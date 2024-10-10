from Utilities import *
import argparse
import yaml
import os
from torch import stack, tensor, set_float32_matmul_precision, Generator, cat, float32, nonzero

set_float32_matmul_precision('high')

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_config(file_path):
    with open(file_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse YAML configuration file')
    
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    
    args = parser.parse_args()

    config = load_config(args.config_file)

    what = config['what']

    if what == 'qm_atomic' or what == 'qm_selection':

        del[config['what']]
        
        out = QM_atomic_pretraining(
            **config
        )
    
    elif what == 'homo-lumo':

        del[config['what']]
        
        out = HLgap_pretraining(
            **config
        )
    
    elif what == 'masking':

        del[config['what']]
        
        out = masking_pretraining(
            **config
        )
    
    elif what == 'tdc':

        del[config['what']]
        
        out = TDC_downstream(
            **config
        )
