import numpy as np
import torch
import pytorch_lightning as pl
import random
from torch.utils.data import TensorDataset, DataLoader
from chytorch.utils.data import MoleculeDataset, collate_molecules, chained_collate, SizedList
from Model import GT
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import Trainer
from tdc.benchmark_group import admet_group
import chython
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import os
from tqdm import tqdm

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def create_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn

def load_n_train(
        batch_size = 32,
        task_name = 'regression',
        nan_token = -10000,
        labels_dict = [{'Y': []}, {'Y': []}],
        packed_mols_list = [[],[]],
        gpu = 'no',
        freeze_encoder = False, 
        checkpoint_path = None, 
        warmup_steps = 5000, 
        unfreeze_step = 1000, 
        dropout = 0.1, 
        d_model = 128, 
        nhead = 16, 
        num_layers = 15, 
        dim_feedforward = 128, 
        lr = 1e-4,
        max_epochs = 2000,
        logs_dir = './generic_logs_dir/',
        log_name = 'generic_log',
        checkpoint_dir = './generic_checkpoint_dir/',
        es_patience = None,
        lr_patience = 30,
        reg_loss = 'l1',
        seed = 42,
        scheduler = None,
        factor = 10,
        cycle_len = 1000,
        qm_atomic_selection = None
):
    set_global_seed(seed)
    worker_init_fn = create_worker_init_fn(seed)
    
    if task_name == 'qm_all':
        labels_dict[0]['nmr'] = [a/100 for a in labels_dict[0]['nmr']]
        labels_dict[1]['nmr'] = [a/100 for a in labels_dict[1]['nmr']]
        data_train = TensorDataset(MoleculeDataset(packed_mols_list[0], unpack=True), SizedList(labels_dict[0]['charges']), SizedList(labels_dict[0]['fukui_e']), SizedList(labels_dict[0]['fukui_n']), SizedList(labels_dict[0]['nmr']))
        data_val = TensorDataset(MoleculeDataset(packed_mols_list[1], unpack=True), SizedList(labels_dict[1]['charges']), SizedList(labels_dict[1]['fukui_e']), SizedList(labels_dict[1]['fukui_n']), SizedList(labels_dict[1]['nmr']))
        dltr = DataLoader(data_train, collate_fn = chained_collate(collate_molecules, torch.cat, torch.cat, torch.cat, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
        dlval = DataLoader(data_val, collate_fn = chained_collate(collate_molecules, torch.cat, torch.cat, torch.cat, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
    
    elif task_name == 'qm_selection':
        if 'nmr' in qm_atomic_selection:
            labels_dict[0]['nmr'] = [a/100 for a in labels_dict[0]['nmr']]
            labels_dict[1]['nmr'] = [a/100 for a in labels_dict[1]['nmr']]
        data_train = TensorDataset(MoleculeDataset(packed_mols_list[0], unpack=True), *[SizedList(labels_dict[0][p]) for p in qm_atomic_selection])
        data_val = TensorDataset(MoleculeDataset(packed_mols_list[1], unpack=True), *[SizedList(labels_dict[1][p]) for p in qm_atomic_selection])
        dltr = DataLoader(data_train, collate_fn = chained_collate(collate_molecules, torch.cat, torch.cat, torch.cat, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
        dlval = DataLoader(data_val, collate_fn = chained_collate(collate_molecules, torch.cat, torch.cat, torch.cat, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
    
    elif task_name == 'qm_one':
        if list(labels_dict[0].keys())[0] == 'nmr':
            labels_dict[0]['nmr'] = [a/100 for a in labels_dict[0]['nmr']]
            labels_dict[1]['nmr'] = [a/100 for a in labels_dict[1]['nmr']]
        data_train = TensorDataset(MoleculeDataset(packed_mols_list[0], unpack=True), SizedList(labels_dict[0][list(labels_dict[0].keys())[0]]))
        data_val = TensorDataset(MoleculeDataset(packed_mols_list[1], unpack=True), SizedList(labels_dict[1][list(labels_dict[1].keys())[0]]))
        dltr = DataLoader(data_train, collate_fn = chained_collate(collate_molecules, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
        dlval = DataLoader(data_val, collate_fn = chained_collate(collate_molecules, torch.cat), shuffle=True, batch_size=batch_size, num_workers=16)
    
    elif task_name == 'masking':
        data_train = TensorDataset(MoleculeDataset(packed_mols_list[0], unpack=True))
        data_val = TensorDataset(MoleculeDataset(packed_mols_list[1], unpack=True))
        dltr = DataLoader(data_train, collate_fn = chained_collate(collate_molecules), shuffle=True, batch_size=batch_size, num_workers=16)
        dlval = DataLoader(data_val, collate_fn = chained_collate(collate_molecules), shuffle=True, batch_size=batch_size, num_workers=16)
    
    elif task_name == 'regression' or task_name == 'classification' or task_name == 'homo-lumo':
        data_train = TensorDataset(MoleculeDataset(packed_mols_list[0], unpack=True), torch.tensor(labels_dict[0]['Y']))
        data_val = TensorDataset(MoleculeDataset(packed_mols_list[1], unpack=True), torch.tensor(labels_dict[1]['Y']))
        dltr = DataLoader(data_train, collate_fn = chained_collate(collate_molecules, torch.stack), shuffle=True, batch_size=batch_size, num_workers=16)
        dlval = DataLoader(data_val, collate_fn = chained_collate(collate_molecules, torch.stack), shuffle=True, batch_size=batch_size, num_workers=16)
    
    if gpu=='no':
    
        acc = 'cpu'
        dev = 1
        strat = 'auto'
    
    else:
        acc = 'gpu'
        dev = gpu
        
        if isinstance(dev, list):
            
            if len(dev)>1:
                strat = 'ddp'
            
            else:
                strat = 'auto'
        
        else:
            
            strat = 'auto'
    
    model = GT(
        task_name=task_name,
        nan_token=nan_token,
        freeze_encoder=freeze_encoder, 
        checkpoint_path=checkpoint_path, 
        warmup_steps = warmup_steps, 
        unfreeze_step = unfreeze_step, 
        dropout = dropout, 
        shared_weights = False, 
        d_model = d_model, 
        nhead = nhead, 
        num_layers = num_layers, 
        dim_feedforward = dim_feedforward, 
        norm_first = True, 
        post_norm = True, 
        zero_bias = True, 
        lr = lr,
        lr_patience = lr_patience,
        reg_loss = reg_loss,
        scheduler = scheduler,
        cycle_len = cycle_len,
        factor = factor
    )
    
    if checkpoint_path is not None:
        check = torch.load(checkpoint_path, map_location = model.device)
        sd = check['state_dict']
        model.encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith('encoder.')})
    
    logger = TensorBoardLogger("{}".format(logs_dir), name="{}".format(log_name))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=0.00, patience=es_patience, verbose=True, mode="min")
    
    checkpoint = ModelCheckpoint(
        dirpath="{}".format(checkpoint_dir),
        monitor='validation_loss',
        save_last=True, 
        save_weights_only=True,
        filename='epoch_{epoch:02d}',
        auto_insert_metric_name=False,
        save_top_k=1
    )
    
    tr = Trainer(
        precision='16-mixed',
        accelerator=acc, 
        devices=dev, 
        strategy = strat, 
        max_epochs = max_epochs, 
        callbacks=[early_stop_callback, lr_monitor, checkpoint], 
        logger = logger, 
        num_sanity_val_steps=1,
        gradient_clip_val=.5
        )

    tr.fit(model, dltr, dlval)
    return model

def get_test_predictions(model, test_structures, device = 'cuda'):
    model.to(device)
    data_test = TensorDataset(MoleculeDataset(test_structures, unpack=True))
    dlts = DataLoader(data_test, collate_fn = chained_collate(collate_molecules), shuffle=False, batch_size=1, num_workers=16)
    predictions = []
    for batch in tqdm(dlts):
        if device == 'cuda':
            p = model.net(batch[0].cuda())
        else:
            p = model.net(batch[0].cpu())
        predictions.append(p)
    predictions = torch.cat(predictions, axis = 0).view(-1).detach().cpu()
    return np.array(predictions)

def QM_atomic_pretraining(
    seed = 42,
    batch_size = 32,
    gpu = 'no', 
    dropout = 0.1, 
    d_model = 128, 
    nhead = 16, 
    num_layers = 15, 
    dim_feedforward = 128, 
    lr = 1e-4,
    max_epochs = 2000,
    data_location = '../qm137/',
    valid_fraction = .1,
    properties = 'all',
    nan_token = -10000,
    logname = None,
    reg_loss = 'l1',
    scheduler = None,
    lr_patience = 100,
    es_patience = 500,
    factor = 10,
    logsdir = './pretraining_logs',
    checkpoint_root = './pretraining_checkpoints',
    qm_atomic_selection = None
    
):
    
    set_global_seed(seed)
    
    all_props = ['nmr', 'fukui_e', 'fukui_n', 'charges']

    if properties == 'all':
        properties = all_props
        task_name = 'qm_all'
    else:
        if properties in all_props:
            properties = [properties]
            task_name = 'qm_one'
        else:
            raise(ValueError, f'{properties} not in the available ones, provide one of: [nmr, fukui_e, fukui_n, charges]')

    with open(data_location + '/structures.pkl', 'rb') as f:
        s = pickle.load(f)
    
    total_len = len(s)
    train_len = int((1-valid_fraction)*total_len)

    tr_indices = set(random.sample(range(total_len), train_len))
    valid_indices = list(set(range(total_len)) - tr_indices)
    tr_indices = list(tr_indices)

    s_train = [s[i] for i in tr_indices]
    s_valid = [s[i] for i in valid_indices]

    structures = [s_train, s_valid]
    
    properties_dict_tr = {}
    properties_dict_valid = {}
    
    for p in properties:

        with open(data_location + f'/{p}.pkl', 'rb') as f:
            pr = pickle.load(f)
        
        pr_tr = [torch.nan_to_num(torch.tensor(pr[i]), nan = nan_token) for i in tr_indices]
        pr_valid = [torch.nan_to_num(torch.tensor(pr[i]), nan = nan_token) for i in valid_indices]

        properties_dict_tr[p] = pr_tr
        properties_dict_valid[p] = pr_valid

    properties_dict = [properties_dict_tr, properties_dict_valid]
    
    if logname == None:
    
        if task_name == 'qm_all':
            logname = 'all'
        else:
            logname = properties[0]

    model = load_n_train(
            batch_size = batch_size,
            task_name = task_name,
            nan_token = nan_token,
            labels_dict = properties_dict,
            packed_mols_list = structures,
            gpu = gpu,
            freeze_encoder = False, 
            checkpoint_path = None, 
            warmup_steps = -1, 
            unfreeze_step = -1, 
            dropout = dropout, 
            d_model = d_model, 
            nhead = nhead, 
            num_layers = num_layers, 
            dim_feedforward = dim_feedforward, 
            lr = lr,
            max_epochs = max_epochs,
            logs_dir = logsdir,
            log_name = f'{logname}',
            checkpoint_dir = f'{checkpoint_root}/qm_{logname}/',
            reg_loss = reg_loss,
            scheduler = scheduler,
            lr_patience = lr_patience,
            factor = factor,
            es_patience = es_patience
    )

    return model


def masking_pretraining(
    seed = 42,
    batch_size = 32,
    gpu = 'no', 
    dropout = 0.1, 
    d_model = 128, 
    nhead = 16, 
    num_layers = 15, 
    dim_feedforward = 128, 
    lr = 1e-4,
    max_epochs = 2000,
    data_location = '../qm137/',
    valid_fraction = .1,
    nan_token = -10000,
    logname = None,
    scheduler = None,
    logsdir = './pretraining_logs',
    checkpoint_root = './pretraining_checkpoints',
    lr_patience = 100,
    es_patience = 500,
    factor = 10,
):
    
    set_global_seed(seed)
    task_name = 'masking'
    with open(data_location + '/structures.pkl', 'rb') as f:
        s = pickle.load(f)
    
    total_len = len(s)
    train_len = int((1-valid_fraction)*total_len)

    tr_indices = set(random.sample(range(total_len), train_len))
    valid_indices = list(set(range(total_len)) - tr_indices)
    tr_indices = list(tr_indices)

    s_train = [s[i] for i in tr_indices]
    s_valid = [s[i] for i in valid_indices]

    structures = [s_train, s_valid]
    
    if logname == None:
        logname = 'masking'
    else:
        pass
    
    model = load_n_train(
            batch_size = batch_size,
            task_name = task_name,
            packed_mols_list = structures,
            gpu = gpu,
            freeze_encoder = False, 
            checkpoint_path = None, 
            warmup_steps = -1, 
            unfreeze_step = -1, 
            dropout = dropout, 
            d_model = d_model, 
            nhead = nhead, 
            num_layers = num_layers, 
            dim_feedforward = dim_feedforward, 
            lr = lr,
            max_epochs = max_epochs,
            logs_dir = logsdir,
            log_name = logname,
            checkpoint_dir = f'{checkpoint_root}/{logname}/',
            scheduler = scheduler,
            lr_patience = lr_patience,
            factor = factor,
            es_patience = es_patience
    )

    return model


def HLgap_pretraining(
    seed = 42,
    batch_size = 32,
    gpu = 'no', 
    dropout = 0.1, 
    d_model = 128, 
    nhead = 16, 
    num_layers = 15, 
    dim_feedforward = 128, 
    lr = 1e-4,
    max_epochs = 2000,
    data_location = '../gap/',
    valid_fraction = .1,
    nan_token = -10000,
    reg_loss = 'l1',
    es_patience = 500,
    lr_patience = 100,
    factor = 10,
    logname = None,
    scheduler = None,
    logsdir = './pretraining_logs',
    checkpoint_root = './pretraining_checkpoints'
):
    
    set_global_seed(seed)
    
    task_name = 'homo-lumo'

    with open(data_location + '/hlgaps.pkl', 'rb') as f:
        pr = pickle.load(f)

    with open(data_location + '/structures.pkl', 'rb') as f:
        s = pickle.load(f)

    where_not_nan = np.argwhere(~np.isnan(pr)).reshape(-1).tolist()
    pr = [pr[i] for i in where_not_nan]
    s = [s[i] for i in where_not_nan]
    
    total_len = len(s)
    train_len = int((1-valid_fraction)*total_len)
    
    tr_indices = set(random.sample(range(total_len), train_len))
    valid_indices = list(set(range(total_len)) - tr_indices)
    tr_indices = list(tr_indices)

    s_train = [s[i] for i in tr_indices]
    s_valid = [s[i] for i in valid_indices]

    structures = [s_train, s_valid]
    
    properties_dict_tr = {}
    properties_dict_valid = {}
        
    pr_tr = [pr[i] for i in tr_indices]
    pr_valid = [pr[i] for i in valid_indices]

    properties_dict_tr['Y'] = pr_tr
    properties_dict_valid['Y'] = pr_valid

    properties_dict = [properties_dict_tr, properties_dict_valid]
    
    if logname is None:
        logname = 'homo-lumo'
    
    model = load_n_train(
            batch_size = batch_size,
            task_name = task_name,
            nan_token = nan_token,
            labels_dict = properties_dict,
            packed_mols_list = structures,
            gpu = gpu,
            freeze_encoder = False, 
            checkpoint_path = None, 
            warmup_steps = -1, 
            unfreeze_step = -1, 
            dropout = dropout, 
            d_model = d_model, 
            nhead = nhead, 
            num_layers = num_layers, 
            dim_feedforward = dim_feedforward, 
            lr = lr,
            max_epochs = max_epochs,
            logs_dir = logsdir,
            log_name = logname,
            checkpoint_dir = f'./{checkpoint_root}/{logname}/',
            reg_loss = reg_loss,
            es_patience = es_patience,
            scheduler = scheduler,
            lr_patience = lr_patience,
            factor = factor
    )
    return model


def check_valid(sm):
    try:
        chython.smiles(sm).pack()
        return True
    except:
        print(sm, 'IS INVALID')
        return False
    

def TDC_downstream(
    seeding = 42,
    task_indices = [],
    seeds = [],
    batch_size = 32,
    gpu = 'no',
    freeze_encoder = False, 
    checkpoint_path = None, 
    warmup_steps = 5000, 
    unfreeze_step = 1000, 
    dropout = 0.1, 
    d_model = 128, 
    nhead = 16, 
    num_layers = 15, 
    dim_feedforward = 128, 
    lr = 1e-4,
    max_epochs = 2000,
    name_run = '',
    es_patience = 200,
    factor = 10,
    lr_patience = 20,
    reg_loss = 'l1',
    scheduler = None,
    logsdir = './TDC_logs',
    checkpoint_root = './TDC_checkpoints'
):
    
    set_global_seed(seeding)
    group = admet_group(path = '../data_tdc/')

    if len(task_indices) == 0:
        names = group.dataset_names
    else:
        names = [group.dataset_names[i] for i in task_indices]

    if len(seeds) == 0:
        seeds = [1, 2, 3, 4, 5]
    else:
        seeds = seeds

    if len(name_run) == 0:
        current_datetime = datetime.now()
        date_time_stamp = current_datetime.strftime("%Y_%m_%d__%H_%M_%S/")
        name_run = date_time_stamp

    results_dict = {}

    for name in names:
        
        predictions_list = []
        
        for seed in seeds:

            benchmark = group.get(name) 

            predictions = {}
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
            
            train['is_valid'] = train['Drug'].apply(lambda x: check_valid(x))
            valid['is_valid'] = valid['Drug'].apply(lambda x: check_valid(x))
            test['is_valid'] = test['Drug'].apply(lambda x: check_valid(x))
            train = train[train['is_valid']]
            valid = valid[valid['is_valid']]
            test = test[test['is_valid']]
            
            train_str = [chython.smiles(sm).pack() for sm in train['Drug']]
            val_str = [chython.smiles(sm).pack() for sm in valid['Drug']]
            test_str = [chython.smiles(sm).pack() for sm in test['Drug']]

            train_lbls = train['Y'].values.tolist()
            val_lbls = valid['Y'].values.tolist()
            test_lbls = test['Y'].values.tolist()

            if len(train['Y'].unique()) == 2:
                task_name = 'classification'
                print("classification")
                scaler = None

            else:
                task_name = 'regression'
                print("regression")
                scaler = StandardScaler()
                scaler.fit(train[['Y']])
                train_lbls = scaler.transform(np.array([train_lbls]).T).tolist()
                val_lbls = scaler.transform(np.array([val_lbls]).T).tolist()
            
            model = load_n_train(
                batch_size = batch_size,
                task_name = task_name,
                labels_dict = [{'Y': train_lbls}, {'Y': val_lbls}],
                packed_mols_list = [train_str, val_str],
                gpu = gpu,
                freeze_encoder = freeze_encoder, 
                checkpoint_path = checkpoint_path, 
                warmup_steps = warmup_steps, 
                unfreeze_step = unfreeze_step, 
                dropout = dropout, 
                d_model = d_model, 
                nhead = nhead, 
                num_layers = num_layers, 
                dim_feedforward = dim_feedforward, 
                lr = lr,
                max_epochs = max_epochs,
                logs_dir = f'{logsdir}/{name_run}/',
                log_name = f'{name}_{seed}',
                checkpoint_dir = f'{checkpoint_root}/{name_run}/{name}_{seed}/',
                es_patience = es_patience,
                lr_patience = lr_patience,
                reg_loss = reg_loss,
                scheduler = scheduler,
                factor = factor
            )

            best_checkpoint_dir = f'{checkpoint_root}/{name_run}/{name}_{seed}/'
            ckpts = os.listdir(best_checkpoint_dir)
            best = [c for c in ckpts if c.startswith('epoch')][0]
            model = model.load_from_checkpoint(best_checkpoint_dir + best)

            model.eval()
            model.freeze()
            y_pred_test = get_test_predictions(model, test_str)

            if scaler is not None:
                y_pred_test = scaler.inverse_transform(np.array([y_pred_test]).T)
                scaler = None

            predictions[name] = y_pred_test

            with open(f'{checkpoint_root}/{name_run}/{name}_{seed}/predictions.pkl', 'wb') as f:
                pickle.dump(y_pred_test, f)

            predictions_list.append(predictions)

        results = group.evaluate_many(predictions_list)
        for k,v in results.items():
            results_dict[k] = v

    with open(f'./results_dicts/{name_run}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    return results_dict
