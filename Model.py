from chytorch.nn import MoleculeEncoder
from chytorch.zoo.utils import pass_suitable_args
from torch import set_float32_matmul_precision, cat, nonzero
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from torch.nn import Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch
import torch.nn as nn

set_float32_matmul_precision('high')

def binary_label_smoothing(true_labels, smoothing=0.25):
    smoothed_labels = true_labels * (1 - smoothing) + 0.5 * smoothing
    return smoothed_labels

CELoss = torch.nn.CrossEntropyLoss()

def LpCensoredLossClassification(target, prediction, p = 1, freq_pos=0.5):
    alpha = 1/freq_pos
    beta = 1/(1-freq_pos)
    alpha_n = alpha/(alpha+beta)
    beta_n = beta/(alpha+beta)
    l1_losses = torch.abs(target - prediction)**p
    l1_losses[target == 1] = alpha_n*l1_losses[target == 1]
    l1_losses[target == 0] = beta_n*l1_losses[target == 0]
    mask_pos = (target == 1) & (prediction >= 1)
    mask_neg = (target == 0) & (prediction <= 0)
    mask = mask_pos | mask_neg  
    l1_losses[mask] = 0
    return torch.mean(l1_losses)

def check_embedding_rename_key(key):
    if key.startswith('embedding.'):
        key = key[len('embedding.'):]
    return key


class GT(LightningModule):

    def __init__(self, 
                 checkpoint_path = None,
                 freeze_encoder = False,
                 warmup_steps = 2500,
                 unfreeze_step = 500,
                 task_name = 'regression',
                 nan_token = -10000,
                 lr = 1e-4,
                 lr_patience = 30,
                 harmonize = False,
                 reg_loss = 'l2',
                 scheduler = None,
                 cycle_len = 1000,
                 factor = 10,
                 random_layers_sampling = False,
                 **kwargs
                 ):
        
        super().__init__()
        self.random_layers_sampling = random_layers_sampling
        self.lr = lr
        self.factor = factor
        self.reg_loss = reg_loss
        self.lr_patience = lr_patience
        self.unfreeze_step = unfreeze_step
        d_model = kwargs.get('d_model', 1024)
        self.encoder = pass_suitable_args(MoleculeEncoder, kwargs)
        self.save_hyperparameters(kwargs)
        self.task_name = task_name
        self.warmup_steps = warmup_steps + unfreeze_step
        self.scheduler = scheduler
        self.cycle_len = cycle_len
        self.layers = kwargs.get('num_layers', 20)
        
        if checkpoint_path is not None:
            check = torch.load(checkpoint_path, map_location = self.device)
            sd = check['state_dict']
            self.encoder.load_state_dict({check_embedding_rename_key(k[8:]): v for k, v in sd.items() if k.startswith('encoder.')})

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        else:
            self.unfreeze_step = -1
            self.warmup_steps = warmup_steps
        
        self.nan_token = nan_token
        
        self.lr = lr
        
        if task_name == 'masking':
            self.head = Linear(d_model, 121)
        
        elif task_name == 'qm_all':
            
            self.head = Linear(d_model, 4)
            
        elif task_name == 'qm_single':
            self.head = Linear(d_model, 1)
        
        elif task_name == 'homo-lumo':
            self.head = Linear(d_model, 1)

        else:
            self.head = Linear(d_model, 1)
    
        self.save_hyperparameters(kwargs)
        
    def net(self, m):
        x = self.encoder(m)
        x_m = x[:,0]
        return self.head(x_m)
        
    def forward_atomic(self, m, head):
        x = self.encoder(m)
        ap = head(x[m.atoms > 2])
        return ap
    
    def forward_masked(self, m, head):
        x = self.encoder(m)
        ap = head(x[m[0] == 2])
        return ap
    
    def forward_bond(self, m, head):
        bi, ni, mi = nonzero(m.distances == 3, as_tuple=True)
        x = self.encoder(m)
        bp = head(cat([x[bi, ni], x[bi, mi]], 1))
        return bp
    
    def forward_mol(self, m, head):
        x = self.encoder(m)
        x_m = x[:,0]
        return head(x_m)
    
    def downstream_predict(self, batch):
        return self.net(batch[0])
    
    def get_cls_token(self, m):
        x = self.encoder(m)
        x_m = x[:,0]
        return x_m
    
    def _step(self, batch, stage):
        
        if stage == 'train':
            
            if self.trainer.global_step == self.unfreeze_step:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = True            
                    
        if self.task_name == 'masking':
            a, n, d = batch[0]
            a_ = a.clone()
            mask = torch.rand_like(a_, dtype=torch.float32) < .15
            mask &= (a_ > 2)
            a_[mask] = 2
            pred = self.forward_masked((a_, n, d), self.head)
            loss = CELoss(pred, a[mask].type(torch.LongTensor).cuda())
            
            self.log('{}_loss_masking'.format(stage), loss.item(), on_step = True, on_epoch = True, prog_bar = True, sync_dist = True, add_dataloader_idx=False)
            total_loss = loss
            self.log(f'{stage}_loss', total_loss)
            
        elif self.task_name == 'qm_all':
            m, *ap = batch
            ap = [a.view(-1,1) for a in ap]
            ap = torch.cat(ap, dim = 1)
            preds = self.forward_atomic(m, self.head)
            total_loss = 0.
            log_loss = 0.
            for i in range(ap.size(-1)):
                
                mask = ap[:,i]!=self.nan_token
                
                if self.reg_loss=='l2':
                    l = (preds[:,i] - ap[:,i])**2
                    
                elif self.reg_loss=='l1':
                    l = (preds[:,i] - ap[:,i]).abs()
                    
                loss = l[mask].mean()
                self.log(f'{stage}_loss_{i}', loss)
                
                log_loss += loss
                
                total_loss += loss
                
            total_loss = total_loss/ap.size(-1)
            self.log(f'{stage}_loss', log_loss/ap.size(-1))
                
        
        elif self.task_name == 'qm_one':
            m, ap = batch
            preds = self.forward_atomic(m, self.head)
            mask = ap!=self.nan_token
            
            if self.reg_loss=='l2':
                l = (preds.view(-1) - ap)**2
                    
            elif self.reg_loss=='l1':
                l = (preds.view(-1) - ap).abs()
                
            loss = l[mask].mean()
            total_loss = loss
            self.log(f'{stage}_loss', total_loss)
        
        elif self.task_name == 'homo-lumo':
            m, mp = batch
            preds = self.forward_mol(m, self.head)
            mask = mp!=self.nan_token
            
            if self.reg_loss=='l2':
                l = (preds.view(-1) - mp)**2
                    
            elif self.reg_loss=='l1':
                l = (preds.view(-1) - mp).abs()
                    
            loss = l[mask].mean()
            total_loss = loss
            self.log(f'{stage}_loss', total_loss)

        elif self.task_name == 'regression':
            m, mp = batch
            preds = self.forward_mol(m, self.head)
            mask = mp!=self.nan_token
            
            if self.reg_loss=='l2':
                l = (preds - mp)**2
                    
            elif self.reg_loss=='l1':
                l = (preds - mp).abs()
                    
            loss = l[mask].mean()
            total_loss = loss
            self.log(f'{stage}_loss', total_loss)
        
        elif self.task_name == 'classification':
            m, mp = batch
            preds = self.forward_mol(m, self.head)
            mask = mp!=self.nan_token
            loss = LpCensoredLossClassification(mp[mask], preds.view(-1)[mask])
            total_loss = loss
            self.log(f'{stage}_loss', total_loss)
            
        if stage == 'train':
            if self.trainer.global_step < self.warmup_steps and self.trainer.global_step >= self.unfreeze_step:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
                for pg in self.optimizers().param_groups:
                    pg["lr"] = lr_scale * self.lr

        return total_loss

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch, _):
        return self._step(batch, 'validation')

    def configure_optimizers(self):
        o = AdamW(self.parameters(), lr=self.lr)
        
        if self.scheduler == 'cycle':
            s = CyclicLR(o, self.lr/self.factor, self.lr, self.cycle_len, mode='triangular', cycle_momentum=False)
            return [o], [{'scheduler': s, 'interval': 'step', 'name': 'lr_scheduler'}]
        
        if self.scheduler == 'const':
            return [o]
        
        else:
            s = ReduceLROnPlateau(o, factor = 0.5, patience = self.lr_patience)
            return [o], [{'scheduler': s, 'name': 'lr_scheduler', 'monitor': 'validation_loss'}]
