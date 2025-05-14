import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# imported modules
import warnings
warnings.filterwarnings("ignore")
import logging
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
# my modules
from model.Module.Dataset import *
from model.Module.Models import *
from model.Module.Runs import *
from model.Module.Utils import *
from model.run_scr.banner import print_banner
from model.run_scr.run_cali import run_cali


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

if __name__ == '__main__':
    print_banner() # print my banner
    mp.freeze_support()  # Needed for Windows
    root_dir = project_root
    # Default configuration and search space settings
    Configs = {
        "input_dim": 6,
        "output_dim": 1,
        "seq_len": 360,
        "shift_size": 180,
        "padding": True,
        "seed": 100,
        "use_gpu": True,
        "exper": "exper2",  # Experiment type ('exper1', 'exper2', or 'exper3'), by default 'exper1'.
        "method": "trans_PIDL",
        "physics_informed": False,
        "pretrain_scale": -1, # number of source lakes for pretraining, maximum: 40, useless during fine-tuning
        "fine_tune_size" : -1, # number or fraction of points to retain for fine-tuning (< 0 for all, 0-1 for percentage, >= 1 absolute number)
        "iter_seed": 0, # seed to fix the random sampling of pretrain_scale and fine_tune_size, useless during fine-tuning
        "verbose": 1,
        "sampler": "products",
        "num_params": 10,
        "parallel": False, #
        "num_of_process": 1, # Number of parallel processes to use for tuning
        "freeze": False # whether to freeze lstm layers
    }
    Search_space = {
        "Epoch": [400],
        "batch_size": [128],
        "learning_rate": [5e-3],
        "hidden_dim": [48],
        "layer_num": [2],
        "hidden_dim_fc1": [32],
        "hidden_dim_fc2": [12],
        "lam_ec": [0]
    }
    ground_data_dir = os.path.join(root_dir, "data", "ccilakes")
    output_base_dir = os.path.join(root_dir, "result", Configs['method'], 'pretrain')
    if not os.path.exists(output_base_dir): os.makedirs(output_base_dir)
    expers = ['exper1', 'exper2', 'exper3']
    # expers = ['exper1']
    pre_train_scales = np.arange(2, 40+1, 2).tolist()
    iter_seeds = [11, 22, 33, 44, 66]
    phases = ['Frigid', 'Cool', 'Temperate', 'Warm', 'Hot']
    for exper in expers:
        if exper == 'exper1':
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "exper1_source.csv"))
            for ps in pre_train_scales:
                if ps >=40 : iter_seeds = [0]
                for iter_seed in iter_seeds:
                    # Set defaults for directories if not provided
                    output_dir = os.path.join(output_base_dir, exper, f'ps_{ps}', f'iter_seed_{iter_seed}')
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    if not os.path.exists(os.path.join(output_dir, 'cal_model.pt')):
                        data_pack = get_pretrain_data(task_list.LakeID.values, ground_data_dir, exper, pretrain_scale=ps, iter_seed=iter_seed)
                        CV_result, cal_result, cal_model, cal_generations = run_cali(Configs, Search_space, data_pack, pretrain_model=None)
                        ## save finetune
                        cal_result.to_csv(os.path.join(output_dir, "cal_result.csv"), index=False)
                        save_model(cal_model, -1, os.path.join(output_dir, "cal_model.pt"))
                        CV_result.to_csv(os.path.join(output_dir, "CV_result.csv"), index=False)
                        # cal_generations.to_csv(os.path.join(output_dir, "cal_generations.csv"), index=False)
        elif exper == 'exper2':
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "exper2_source.csv"))
            output_dir = os.path.join(output_base_dir, exper)
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir, 'cal_model.pt')):
                data_pack = get_pretrain_data(task_list.LakeID.values, ground_data_dir, exper, pretrain_scale=-1,iter_seed=0)
                CV_result, cal_result, cal_model, cal_generations = run_cali(Configs, Search_space, data_pack, pretrain_model=None)
                ## save finetune
                cal_result.to_csv(os.path.join(output_dir, "cal_result.csv"), index=False)
                save_model(cal_model, -1, os.path.join(output_dir, "cal_model.pt"))
                CV_result.to_csv(os.path.join(output_dir, "CV_result.csv"), index=False)
                # cal_generations.to_csv(os.path.join(output_dir, "cal_generations.csv"), index=False)

        else:
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "ESA_CCI.csv"))
            for phase in phases:
                lake_list = task_list.query('Phase == @phase').reset_index(drop=True)
                lake_list = lake_list.sort_values(by=['Cover_1995_2020'], ascending=False).reset_index(drop=True)[:40]
                lake_list = lake_list.sort_values(by=['LakeID'], ascending=True).reset_index(drop=True)
                output_dir = os.path.join(output_base_dir, exper, phase)
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                if not os.path.exists(os.path.join(output_dir, 'cal_model.pt')):
                    data_pack = get_pretrain_data(lake_list.LakeID.values, ground_data_dir, exper, pretrain_scale=-1,iter_seed=0)
                    CV_result, cal_result, cal_model, cal_generations = run_cali(Configs, Search_space, data_pack, pretrain_model=None)
                    ## save finetune
                    cal_result.to_csv(os.path.join(output_dir, "cal_result.csv"), index=False)
                    save_model(cal_model, -1, os.path.join(output_dir, "cal_model.pt"))
                    CV_result.to_csv(os.path.join(output_dir, "CV_result.csv"), index=False)
                    # cal_generations.to_csv(os.path.join(output_dir, "cal_generations.csv"), index=False)
