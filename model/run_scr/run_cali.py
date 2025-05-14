# required for console runs
import shutil
term_width = shutil.get_terminal_size((80, 20)).columns - 40  # subtract logger length
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# imported modules
import warnings
warnings.filterwarnings("ignore")
import logging
import json
import pandas as pd
import multiprocessing as mp
# my modules
from model.Module.Dataset import *
from model.Module.Models import *
from model.Module.Runs import *
from model.Module.Utils import *

# Set environment variables for controlled threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Set up logger without basicConfig to avoid default handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Add custom flushable console handler
console_handler = FlushableStreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def run_task(arg):
    (Configs, Search_space, data_pack, pretrain_model) = arg
    ###--------------------------------console print only--------------------------------###
    term_width = shutil.get_terminal_size((80, 20)).columns - 40  # subtract logger length
    border = "-" * term_width
    prefix = " "
    suffix = " "
    content_width = term_width - len(prefix) - len(suffix)
    ###--------------------------------console print only--------------------------------###
    logger.info(border)
    logger.info(f"{prefix}{f'Starting training for {Configs['method']}'.center(content_width)}{suffix}")
    logger.info(border)
    ###--------------------------------console print only--------------------------------###
    # data pack orders [Phys, ground_labels, distill_labels]
    [train_data_pack, test_data_pack] = data_pack
    # self-update model configs and search space based on usr's situation
    Configs_updated = Configs
    Configs_updated |= dict(input_dim=test_data_pack[0].shape[-1])
    Configs_updated |= dict(device=get_device(Configs.get('use_gpu', False)))
    #
    Search_space_updated = Search_space
    ###--------------------------------console print only--------------------------------###
    logger.info(border)
    logger.info(f"{prefix}{f'Calibrating model for {Configs['method']}'.center(content_width)}{suffix}")
    logger.info(f"{prefix}{f'{Configs['num_of_process']} processes in parallel'.center(content_width)}{suffix}")
    logger.info(border)
    ###--------------------------------console print only--------------------------------###
    CV_result, _, _ = cali_mp(Configs_updated, Search_space_updated, train_data_pack, train_data_pack, pretrain_model)
    optim = CV_result.sort_values('train_rmse', ascending=True).iloc[0]
    # model structure update
    optim_hyper = {
        "Epoch": [int(optim['Epoch'])],
        "batch_size": [int(optim['batch_size'])],
        "learning_rate": [optim['learning_rate']],
        "hidden_dim": [int(optim['hidden_dim'])],
        "layer_num": [int(optim['layer_num'])],
        "hidden_dim_fc1": [int(optim['hidden_dim_fc1'])],
        "hidden_dim_fc2": [int(optim['hidden_dim_fc2'])],
        "lam_ec": [optim['lam_ec']]
    }
    # retrain
    ###--------------------------------console print only--------------------------------###
    logger.info(border)
    logger.info(f"{prefix}{f'Retraining model for {Configs['method']}'.center(content_width)}{suffix}")
    logger.info(border)
    ###--------------------------------console print only--------------------------------###
    cal_result, cal_model, cal_generations = cali_mp(Configs_updated, optim_hyper, train_data_pack, test_data_pack, pretrain_model)
    ###--------------------------------console print only--------------------------------###
    logger.info(border)
    logger.info(f"{prefix}{f'Completed parallel training : {Configs['method']}'.center(content_width)}{suffix}")
    logger.info(border)
    ###--------------------------------console print only--------------------------------###

    return CV_result, cal_result, cal_model, cal_generations

def run_cali(Configs, Search_space, data_pack, pretrain_model):
    # torch.set_num_threads(1)
    arg = (Configs, Search_space, data_pack, pretrain_model)
    try:
        CV_result, cal_result, cal_model, cal_generations = run_task(arg)
        cal_result = cal_result.drop(columns='model_idx')
        cal_generations = cal_generations.drop(columns='model_idx')
    except Exception as e:
        print('run_cali failed in main:', e)
        CV_result, cal_result, cal_model, cal_generations = None, None, None, None
    # ###--------------------------------console print only--------------------------------###
    # border = "*" * term_width
    # prefix = " "
    # suffix = " "
    # content_width = term_width - len(prefix) - len(suffix)
    # logger.info(border)
    # logger.info(f"{prefix}{f'file saved to : {output_dir}'.center(content_width)}{suffix}")
    # logger.info(border)
    ###--------------------------------console print only--------------------------------###
    return CV_result, cal_result, cal_model[0], cal_generations