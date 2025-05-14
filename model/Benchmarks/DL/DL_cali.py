import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
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

def get_data(lake_id, ground_data_dir, synthetic_data_dir, exper, require_synthetic_data = False):
    phys_var = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Ice']
    input_var = ['DOY', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed']
    #
    met = pd.read_csv(os.path.join(ground_data_dir, 'mets', f'met_{lake_id}.csv'))
    met['Date'] = pd.to_datetime(met['Date'], format='mixed')
    lswt = pd.read_csv(os.path.join(ground_data_dir, 'lswts', f'lake_id_{lake_id}.csv'))
    lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
    data_set = pd.merge(met, lswt, on='Date', how='inner')
    data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
    data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
    data_set['Year'] = data_set['Date'].dt.year
    #
    if require_synthetic_data:
        synthetic_dir_update = os.path.join(synthetic_data_dir, Configs['exper'], str(lake_id))
        synthetic_data = get_synthetic_label(synthetic_dir_update)
        data_set = pd.merge(data_set, synthetic_data, on='Date', how='inner')
        data_set['LSWT'] = data_set['LSWT_sims'].values
    # Define train and test dataset according to experiment type
    if exper == 'exper1':
        train_sets = data_set.query('Year <= 2014').reset_index(drop=True)
        test_sets = data_set.query('Year > 2014').reset_index(drop=True)
    elif exper == 'exper2':
        demask_df = data_set.query('Mask == 1').reset_index(drop=True)
        train_idx, _ = data_split(demask_df, test_size=0.4, shuffle=False)
        split_date = train_idx.Date.values[-1]
        train_sets = data_set.query('Date <= @split_date').reset_index(drop=True)
        test_sets = data_set.query('Date > @split_date').reset_index(drop=True)
    else:
        train_sets = data_set.query('Year >= 1995 and Year <= 2010').reset_index(drop=True)
        test_sets = data_set.query('Year > 2010').reset_index(drop=True)
    # Package train and test data (features, labels, phys_vars, mask)
    train_data_pack = [train_sets[input_var].values,
                       train_sets['LSWT'].values,
                       train_sets[phys_var].values
                       ]

    test_data_pack = [test_sets[input_var].values,
                      test_sets['LSWT'].values,
                      test_sets[phys_var].values
                      ]

    return [train_data_pack, test_data_pack]


def run_task(Configs, Search_space, data_packs, output_dir):
    # unpack
    Configs |= dict(physics_informed=False)
    Configs |= dict(freeze=False)
    ground_data_pack = data_packs
    CV_result, cal_result, cal_model, cal_generations = run_cali(Configs, Search_space, ground_data_pack, pretrain_model=None)
    ## save finetune
    cal_result.to_csv(os.path.join(output_dir, "cal_result.csv"), index=False)
    save_model(cal_model, -1, os.path.join(output_dir, "cal_model.pt"))
    CV_result.to_csv(os.path.join(output_dir, "CV_result.csv"), index=False)
    cal_generations.to_csv(os.path.join(output_dir, "cal_generations.csv"), index=False)

def run_task_mp(Configs, Search_space, task_list, ground_data_dir, synthetic_data_dir, output_base_dir):
    aggregated_results = []
    for lake_id in task_list:
        output_dir = os.path.join(output_base_dir, str(lake_id))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_file = os.path.join(output_dir, 'cal_result.csv')
        if not os.path.isfile(result_file):
            ground_data_pack = get_data(lake_id, ground_data_dir, synthetic_data_dir,Configs['exper'], require_synthetic_data=False)
            data_packs = ground_data_pack
            run_task(Configs, Search_space, data_packs, output_dir)
        result = pd.read_csv(result_file)
        result['method'] = Configs['method']
        result['exper'] = Configs['exper']
        result['lake_id'] = lake_id
        result['seed'] = Configs['seed']
        aggregated_results.append(result)
        # Concatenate all aggregated results and save to a CSV file
        aggregated_df = pd.concat(aggregated_results, ignore_index=True)
        aggregated_csv_path = os.path.join(output_base_dir, f"{Configs['method']}.csv")
        aggregated_df.to_csv(aggregated_csv_path, index=False)


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
        "exper": "exper1",  # Experiment type ('exper1', 'exper2', or 'exper3'), by default 'exper1'.
        "method": "DL",
        "physics_informed": False,
        "pretrain_scale": -1, # number of source lakes for pretraining, maximum: 40, useless during fine-tuning
        "fine_tune_size" : -1, # number or fraction of points to retain for fine-tuning (< 0 for all, 0-1 for percentage, >= 1 absolute number)
        "iter_seed": 0, # seed to fix the random sampling of pretrain_scale and fine_tune_size, useless during fine-tuning
        "verbose": 1,
        "sampler": "products",
        "num_params": 10,
        "parallel": True, #
        "num_of_process": 10, # Number of parallel processes to use for tuning
        "freeze": False # whether to freeze lstm layers
    }
    Search_space = {
        "Epoch": [200],
        "batch_size": [128],
        "learning_rate": [5e-4, 5e-3 ,5e-2],
        "hidden_dim": [48],
        "layer_num": [2],
        "hidden_dim_fc1": [32],
        "hidden_dim_fc2": [12],
        "lam_ec": [0]
    }
    expers = ['exper1', 'exper2', 'exper3']
    for exper in expers:
        Configs |= dict(exper=exper)
        if exper == 'exper1':
            ground_data_dir = os.path.join(root_dir, "data", "insitu_valid")
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "exper1_target_valid.csv")).Name.values
            synthetic_data_dir = os.path.join(root_dir, "result", "PB")
        elif exper == 'exper2':
            ground_data_dir =  os.path.join(root_dir, "data", "insitulakes")
            synthetic_data_dir = os.path.join(root_dir, "result", "PB")
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "exper2_target.csv")).LakeID.values
        else:
            ground_data_dir = os.path.join(root_dir, "data", "ccilakes")
            synthetic_data_dir = os.path.join(root_dir, "result", "PB")
            task_list = pd.read_csv(os.path.join(ground_data_dir, "lake_info", "ESA_CCI.csv")).LakeID.values
        # Set defaults for directories if not provided
        output_base_dir = os.path.join(root_dir, "result", Configs['method'], Configs['exper'])
        if not os.path.exists(output_base_dir): os.makedirs(output_base_dir)
        run_task_mp(Configs, Search_space, task_list, ground_data_dir, synthetic_data_dir, output_base_dir)