import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from chytorch.utils.data import MoleculeDataset, collate_molecules, chained_collate
from Model import GT
import chython
import os
import copy
from tqdm import tqdm
from scipy.optimize import minimize
from chytorch.utils.data import MoleculeDataset
from chytorch.utils.data import collate_molecules, chained_collate, SMILESDataset
from IPython.display import clear_output
from torch.autograd.functional import jacobian
import random
import pytorch_lightning as pl
import torch.nn as nn


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def l1_inf_norm(X):
    l1_norm = np.linalg.norm(X, ord=1)  
    l_inf_norm = np.linalg.norm(X, ord=np.inf)  
    l1_inf_norm = np.sqrt(l1_norm * l_inf_norm)      
    return l1_inf_norm


def objective_function(v, x):
    w = np.outer(np.ones(len(x)), v)
    return np.linalg.norm(x-w) 

def get_minimal(x):

    x = torch.tensor(x).detach()
    
    initial_guess = x.mean(dim = 0).numpy()
    
    result = minimize(objective_function, initial_guess, args = x.numpy(), tol = 1e-5, method = 'BFGS')

    minimized_value = result.x
    return l1_inf_norm(x.numpy()-np.array(minimized_value))



def get_iterator(molecules, is_prepared_as_packed_chython = False, return_unpacked_molecules = False):
    
    if not is_prepared_as_packed_chython:
        
        molecules_prepared = []
        molecules_unpacked = []
        
        for mol in molecules:
            
            try:
                s = chython.smiles(mol).pack()
                molecules_prepared.append(s)
                molecules_unpacked.append(chython.MoleculeContainer.unpack(s))
            
            except:
                print(mol, "IS NOT VALID")
    else:
        molecules_prepared = molecules
        
                
    data = TensorDataset(MoleculeDataset(molecules_prepared, unpack = True))
    dlts = DataLoader(data, collate_fn=chained_collate(collate_molecules), shuffle=False, batch_size=1)
    
    if return_unpacked_molecules:
        return dlts, molecules_unpacked
    else:
        return dlts    


def get_dtl_from_smiles(smiles):
    
    mol = MoleculeDataset(SMILESDataset(smiles), unpack=False)
    data = TensorDataset(mol)
    dlts = DataLoader(data, collate_fn=chained_collate(collate_molecules), shuffle=False, batch_size=1)
    return dlts


def get_Jacobian_ij(model, batch, i, j, n = None, device = 'cuda'):
    
    batch[0] = batch[0].to(device)
    model.to(device)
    for name, param in model.named_parameters:
        param.requires_grad_(True)
        
    model.eval()
    
    atoms, neighbors, distances = batch[0]
    encoder = model.encoder
    
    xo = encoder.atoms_encoder(atoms) +  encoder.neighbors_encoder(neighbors)
    xo.requires_grad(True)
    
    d_mask = encoder.distance_encoders[0](distances).permute(0, 3, 1, 2).flatten(end_dim=1)
    d_mask.requires_grad(True)
    if n is None:
        n = len(encoder.layers)
    
    layer = encoder.layers[0]
    x, a = layer(xo, d_mask) 
    
    for j in range(1,n):
        
        layer = encoder.layers[j]
        x, a = layer(x, d_mask)  
        
    gradients = [[torch.autograd.grad(outputs=x[0,i,k], inputs=xo[0,j,h], grad_outputs = 1.0, retain_graph=True, create_graph=True).item() for k in range(x.size(-1))] for h in range(xo.size(-1))]
    
    return gradients


def get_rank_residuals(model, batch, n = None, device = 'cuda'):
    
    model.eval()
    
    if n is None:
        n = len( model.encoder.layers)
    
    ranks = []
    
    for i in range(1, n):
        result, a_, x, xo, L = get_nth_layer(model, batch, x_ = True, n = i, device = device, double_cast = False)
        ranks.append(get_minimal(x[0].detach().cpu().numpy())/l1_inf_norm(x[0].detach().cpu().numpy()))
    
    return ranks

def get_metrics(eigvecs_r, eigvecs_l, eigvals_r, t = 0.9):
    
    # Laplacian eigenvectors are N-dimensional with N nodes in the graph, but Rollout comes from an Attention matrix that considers
    # the CLS token as well, so the first degree of freedom of each eigenvector is cut out and the vector is renormalized
    # also for the same reason we have N laplacian eigenvectors and eigenvalues, 
    # so we discard the last eigenvector and eigenvalue from Rollout (it is the least important one for the considered ordering)
    a = np.real(eigvecs_l[1:,:])/np.linalg.norm(np.real(eigvecs_l[1:,:]),axis = 0)
    b = np.real(eigvecs_r[1:,:])/np.linalg.norm(np.real(eigvecs_r[1:,:]), axis = 0)
    
    # we compute the outer product, or overlap matrix C_ij = |<a_i|l_j>| and we do not consider the row and column relative to the trivial eigenvectors
    outer = np.abs(np.einsum('ij,jk->ik', np.conj(b.T), (a)))[1:,1:]
    
    # get the max in each column, namely max_i{C_ij}.
    diag = np.max(outer, axis = 0)
    
    # turn this into a mask of 0 and 1 based on the threshold value
    which_diag = copy.deepcopy(diag)
    
    # get which laplacian modes overlap well
    which_laplacian = np.argwhere(which_diag>t).reshape(-1)
    
    # turn the threshold into a (0,1) mask for the eigenvalues
    which_diag[which_diag<t] = 0.0
    which_diag[which_diag!=0.0] = 1.
    
    # get the eigenvalues and compute \zeta = \eta*number_of_laplacians
    eigvals = np.absolute(eigvals_r[1:])
    fraction = (which_diag*eigvals).sum()/(eigvals.sum())
    quantity = fraction*(which_diag).sum()
    number_of_laplacians = np.sum(which_diag)
    
    # in case something goes wrong
    if np.isnan(quantity):
        print('something went wrong')
        quantity = 0
    
    return quantity, fraction, number_of_laplacians, which_laplacian

def transfer_weights(source_path, target_model, freeze=False, device='cuda'):
    
    source_state_dict = torch.load(source_path, map_location=device)['state_dict']
    
    target_state_dict = target_model.state_dict()
    
    transferred_keys = []
    
    for k, v in source_state_dict.items():
        if k in target_state_dict:
            target_state_dict[k] = v
            transferred_keys.append(k)
        
        if k.replace('encoder.', 'encoder.embedding.', 1) in target_state_dict:
            target_state_dict[k.replace('encoder.', 'encoder.embedding', 1)] = v
            transferred_keys.append(k.replace('encoder.', 'encoder.embedding', 1))
            
        else:
            if k.startswith('net.0.') and k.replace('net.0.', 'encoder.', 1) in target_state_dict:
                new_key = k.replace('net.0.', 'encoder.', 1)
                if v.size() == target_state_dict[new_key].size():
                    target_state_dict[new_key] = v
                    transferred_keys.append(new_key)
                    
            if k.startswith('net.0.') and k.replace('net.0.', 'encoder.embedding.', 1) in target_state_dict:
                new_key = k.replace('net.0.', 'encoder.embedding.', 1)
                if v.size() == target_state_dict[new_key].size():
                    target_state_dict[new_key] = v
                    transferred_keys.append(new_key)
                    
    target_state_dict_2 = target_model.state_dict()
    filtered_state_dict = {k: v for k, v in target_state_dict.items() if k in target_state_dict_2 and v.size() == target_state_dict_2[k].size()}
     
    target_model.load_state_dict(filtered_state_dict, strict=False)
    
    if freeze:
        for name, param in target_model.named_parameters():
            if name in transferred_keys:
                param.requires_grad = False
    
    target_model.to(device)
    
    return target_model, transferred_keys

def get_nth_layer(model, batch, rollout = True, x_ = False, laplacian = True, n = -1, latent = False, device = 'cpu', double_cast = True):
    
    batch[0] = batch[0].to(device)
    model.to(device)
    
    if double_cast:
        model = model.double()
    model.eval()
    model.freeze()
    atoms, neighbors, distances = batch[0]
    
    L = None
    
    if laplacian:
        A = copy.deepcopy(distances[0].float().detach().cpu().numpy())
        #A = A[1:, 1:]
        A[A==2] = 0
        A[A>3] = 0
        A[A==3] = 1
        A[0,0] = 0
        D = np.diag(np.sum(A, axis = 1))
        L = D - A
        
    x = model.encoder.atoms_encoder(atoms) +  model.encoder.neighbors_encoder(neighbors)
    xo = x
    d_mask = model.encoder.distance_encoder(distances).permute(0, 3, 1, 2).flatten(end_dim=1)
    
    a_ = None
    m = model.hparams['nhead']
    
    if n == -1:
        n = model.hparams['num_layers']
    
    for j in range(0,n):
        
        lr = model.encoder.layers[j]
        if double_cast:
            x = x.double()
            d_mask = d_mask.double()
        x, a = lr(x, d_mask, need_weights=rollout)  # noqa
        a = 0.5*(a + torch.eye(a.size(2), device = device).view(1,a.size(1), -1))
        
        if a_ is None:
            a_ = a  
        
        else:
            a_ = torch.bmm(a, a_)
            
    a_= a_.detach().tolist()
    zero_mask = atoms != 0
    zero_mask = zero_mask.to(device)
    #x = x[:,1:,:]
    result = x[zero_mask[:, :]].detach().cpu().numpy()
    #result = result.reshape(result.shape[0]*result.shape[1],result.shape[2]).detach().cpu().numpy()
    if x_:
        return result, a_, x, xo, L
    else:       
        return result, a_, L


def Laplacian_Rollout_analysis(checkpoint_path, molecules, is_prepared_as_packed_chython = False, return_weights = False, device = 'cpu'):
    
    loaded_path_hyper_dict = torch.load(checkpoint_path)['hyper_parameters']
    try:
        model = GT(
            checkpoint_path = checkpoint_path,
            **loaded_path_hyper_dict
        )
    except:
        print('doing the transfer weight thing')
        model = GT(
            checkpoint_path = None,
            **loaded_path_hyper_dict
        )
    
    model, w = transfer_weights(checkpoint_path, model, device = device)
    model.eval()
    model.freeze()
    print(len(w))
    dataloader = get_dtl_from_smiles(molecules)#, is_prepared_as_packed_chython)
    
    if not is_prepared_as_packed_chython:
        
        molecules_prepared = []
        
        for mol in molecules:
            
            try:
                s = chython.smiles(mol).pack()
                molecules_prepared.append(s)
                
            except:
                print(mol, "IS NOT VALID")
        
    molecules = molecules_prepared
    zetas = []
    etas = []
    number_of_laplacians_ = []
    which_laplacians = []
    
    for i, batch in enumerate(dataloader):
        model.eval()
        _, rollout, L = get_nth_layer(model, batch, latent = False, laplacian = True, device = device)
        
        rollout = np.array(rollout[0])
        
        eigvals_r, eigvecs_r = np.linalg.eig(rollout)
        idx_r = np.flip(np.argsort(np.abs(eigvals_r)))
        eigvals_r = eigvals_r[idx_r]
        eigvecs_r = eigvecs_r[:,idx_r]

        eigvals_l, eigvecs_l = np.linalg.eig(L)
        idx_l = np.argsort(eigvals_l)
        eigvals_l = eigvals_l[idx_l]
        eigvecs_l = eigvecs_l[:,idx_l]
        
        zeta, eta, number_of_laplacians, which_laplacian = get_metrics(eigvecs_r, eigvecs_l, eigvals_r, t = 0.9)
        zetas.append(zeta)
        etas.append(eta)
        number_of_laplacians_.append(number_of_laplacians)
        which_laplacians.append(which_laplacian)
    
    if return_weights:
        return zetas, etas, number_of_laplacians_, which_laplacians, w
    else:
        return zetas, etas, number_of_laplacians_, which_laplacians


def get_Jacobian_ij(model, batch, n=None, device='cuda'):
    
    batch[0] = batch[0].to(device)
    model.to(device)
    
    atoms, neighbors, distances = batch[0]
    
    encoder = model.encoder
    
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()
    
    xo = encoder.atoms_encoder(atoms) + encoder.neighbors_encoder(neighbors)
    xo.requires_grad_(True)
    
    d_mask = encoder.distance_encoders[0](distances).permute(0, 3, 1, 2).flatten(end_dim=1)
    d_mask.requires_grad_(True)
    
    def get_output(xo, n = n):
    
        if n is None:
            n = len(encoder.layers)-1

        x = xo
        for k in range(n):
            layer = encoder.layers[k]
            x, _ = layer(x, d_mask)

        return x

    jac = jacobian(get_output, xo, vectorize = True)
    
    return jac

def get_sensitivities_per_topdistance(ckpt_path, batch, depth = None):
    
    ckpts = os.listdir(ckpt_path)
    ckpt = [c for c in ckpts if c.startswith('epoch')][0]

    checkpoint_path = f'{ckpt_path}/{ckpt}'
    
    loaded_path_hyper_dict = torch.load(checkpoint_path)['hyper_parameters']

    model = GT(
        checkpoint_path = checkpoint_path,
        **loaded_path_hyper_dict
    )

    top_dist = batch[0].distances[0].cpu().numpy()
    top_dist = top_dist[1:,1:]
    sensitivities = []
    
    if len(batch[0].atoms[0])>=4 and len(batch[0].atoms[0])<=50:
        
        jacc = get_Jacobian_ij(model, batch, n = depth).squeeze(0).squeeze(2).transpose(1,2)
        jacc = jacc[1:,1:,:,:]
        torch.cuda.empty_cache()

        for k in tqdm(range(2, np.max(top_dist))):
            jac_avgs = []

            for atom_idx in range(0, len(top_dist)):
                atoms_to_check = np.argwhere(top_dist[atom_idx] == k).reshape(-1)
                jac_avg = []

                for j in atoms_to_check:
                    jac = jacc[atom_idx, j].norm().item()
                    jac_avg.append(jac)

                jac_avgs.append(np.mean(jac_avg))
                
            sensitivities.append(np.mean(jac_avgs))

        return sensitivities
    else:
        print('too short or too long')
        print(len(batch[0].atoms[0]))
