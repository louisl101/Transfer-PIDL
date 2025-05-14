#%%
import numpy as np
import pandas as pd
import math
import torch
import random
from sklearn import preprocessing
from torch.utils.data import TensorDataset,DataLoader
import os
import itertools
from tqdm import trange
import multiprocessing as mp

from model.Module.Dataset import *
from model.Module.Runs import *
from model.Module.Utils import *


if __name__ == "__main__":
    input_var = ['DOY', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed']
    root_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    # exper2
    insitu_dir = os.path.join(root_dir, 'data', 'insitulakes')
    mdl_dir = os.path.join(root_dir, 'result', 'PIDL', 'exper1')
    lake_list = pd.read_csv(os.path.join(insitu_dir, 'lake_info', 'exper1_target.csv'))
    output_dir = os.path.join(root_dir, 'result', 'PIDL', 'projection')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    obs_dir = os.path.join(insitu_dir, 'lswts')
    pred_dir = os.path.join(insitu_dir, 'mets')
    for i in trange(len(lake_list)):
        lake_name = lake_list.Name.values[i]
        lake_id = lake_list.LakeID.values[i]
        try:
            # lake_id = 2
            met = pd.read_csv(os.path.join(pred_dir, f'met_{lake_id}.csv'))
            met['Date'] = pd.to_datetime(met['Date'], format='mixed')
            mdl = load_model(os.path.join(mdl_dir, lake_name, f'cal_model.pt'))
            lswt = pd.read_csv(os.path.join(obs_dir, f'lake_id_{lake_id}.csv'))
            lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
            data_set = pd.merge(met, lswt, on='Date', how='inner')
            data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
            data_set['Year'] = data_set['Date'].dt.year
            data_set['Mask'] = 1
            data_set['Mask_obs'] = data_set['LSWT'].notnull().values.astype(int)
            # cali and vali dataset
            cal_set = data_set.query('Year <= 2014').reset_index(drop=True)
            val_set = data_set.query('Year > 2014').reset_index(drop=True)
            cal_data_pack = [cal_set[input_var].values,
                               cal_set['Mask'].values
                               ]

            val_data_pack = [val_set[input_var].values,
                              val_set['Mask'].values
                              ]
            cal_generation = project(360, cal_data_pack, mdl, 100, 'cpu')
            val_generation = project(360, val_data_pack, mdl, 100, 'cpu')
            #
            cal_set['projection'] = cal_generation
            val_set['projection'] = val_generation
            cal_set['Phase'] = 'Calibration'
            val_set['Phase'] = 'Validation'
            projection = pd.concat((
                cal_set[['LSWT','projection','Phase','Year','Date']],
                val_set[['LSWT','projection','Phase','Year','Date']]
            ), axis=0).reset_index(drop=True)
            projection['Error'] = (projection['projection'].values - projection['LSWT'].values) / (
                    abs(projection['LSWT'].values) + abs(projection['projection'].values))

            projection['Lake'] = lake_id
            projection.loc[projection['projection'] < 0, 'projection'] = np.float32(0.01)
            projection.to_csv(os.path.join(output_dir, f'{lake_id}.csv'), index=False)
        except:
            print(lake_id, 'is empty')