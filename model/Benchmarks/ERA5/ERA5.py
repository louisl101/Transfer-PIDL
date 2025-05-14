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
from model.Module.Utils import *


if __name__ == "__main__":
    database = 'MixedTemp'  # 'glast','skintemp','mixlayer'
    root_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    ##
    # exper1
    insitu_dir = os.path.join(root_dir, 'data', 'insitulakes')
    lake_list = pd.read_csv(os.path.join(insitu_dir, 'lake_info', 'exper1_target.csv'))
    obs_dir = os.path.join(insitu_dir,'lswts')
    pred_dir = os.path.join(insitu_dir,'mets')
    performances = []
    for i in trange(len(lake_list)):
        lake_id = lake_list.LakeID.values[i]
        try:
            # lake_id = 2
            met = pd.read_csv(os.path.join(pred_dir, f'met_{lake_id}.csv'))
            met['Date'] = pd.to_datetime(met['Date'], format='mixed')
            lswt = pd.read_csv(os.path.join(obs_dir, f'lake_id_{lake_id}.csv'))
            lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
            data_set = pd.merge(met, lswt, on='Date', how='inner')
            data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
            data_set['LSWT_pred'] = data_set['MixedTemp'].values
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set['Year'] = data_set['Date'].dt.year
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set = data_set.query('Mask==1').reset_index(drop=True)
            #
            cal_set = data_set.query('Year <= 2014').reset_index(drop=True)
            val_set = data_set.query('Year > 2014').reset_index(drop=True)
            train_obs, train_pred = cal_set['LSWT'].values, cal_set['LSWT_pred'].values
            test_obs, test_pred = val_set['LSWT'].values, val_set['LSWT_pred'].values
            #
            train_rmse = RMSE(train_obs, train_pred)
            test_rmse = RMSE(test_obs, test_pred)
            #
            train_r2 = R_squared(train_obs, train_pred)
            test_r2 = R_squared(test_obs, test_pred)
            #
            train_mae = MAE(train_obs, train_pred)
            test_mae = MAE(test_obs, test_pred)
            # metrics calculation
            performance = pd.DataFrame((train_rmse, test_rmse,
                                        train_r2, test_r2,
                                        train_mae, test_mae,
                                        lake_id),
                                       index=[
                                           'train_rmse', 'test_rmse',
                                           'train_r2', 'test_r2',
                                           'train_mae', 'test_mae',
                                           'lake_id']
                                       ).T
            performance['Model'] = database
            performances.append(performance)
        except:
            print(lake_id, 'is empty')
    performances = pd.concat(performances)
    output_dir = os.path.join(root_dir, 'result', 'EAR5', 'exper1')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    performances.to_csv(os.path.join(output_dir, f'{database}.csv'), index=False)

    # exper2
    insitu_dir = os.path.join(root_dir, 'data', 'insitulakes')
    lake_list = pd.read_csv(os.path.join(insitu_dir, 'lake_info', 'exper2_target.csv'))
    obs_dir = os.path.join(insitu_dir, 'lswts')
    pred_dir = os.path.join(insitu_dir, 'mets')
    performances = []
    for i in trange(len(lake_list)):
        lake_id = lake_list.LakeID.values[i]
        try:
            # lake_id = 2
            met = pd.read_csv(os.path.join(pred_dir, f'met_{lake_id}.csv'))
            met['Date'] = pd.to_datetime(met['Date'], format='mixed')
            lswt = pd.read_csv(os.path.join(obs_dir, f'lake_id_{lake_id}.csv'))
            lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
            data_set = pd.merge(met, lswt, on='Date', how='inner')
            data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
            data_set['LSWT_pred'] = data_set['MixedTemp'].values
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set['Year'] = data_set['Date'].dt.year
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set = data_set.query('Mask==1').reset_index(drop=True)
            #
            demask_df = data_set.query('Mask == 1').reset_index(drop=True)
            train_idx, _ = data_split(demask_df, test_size=0.4, shuffle=False)
            split_date = train_idx.Date.values[-1]
            cal_set = data_set.query('Date <= @split_date').reset_index(drop=True)
            val_set = data_set.query('Date > @split_date').reset_index(drop=True)
            #
            train_obs, train_pred = cal_set['LSWT'].values, cal_set['LSWT_pred'].values
            test_obs, test_pred = val_set['LSWT'].values, val_set['LSWT_pred'].values
            #
            train_rmse = RMSE(train_obs, train_pred)
            test_rmse = RMSE(test_obs, test_pred)
            #
            train_r2 = R_squared(train_obs, train_pred)
            test_r2 = R_squared(test_obs, test_pred)
            #
            train_mae = MAE(train_obs, train_pred)
            test_mae = MAE(test_obs, test_pred)
            # metrics calculation
            performance = pd.DataFrame((train_rmse, test_rmse,
                                        train_r2, test_r2,
                                        train_mae, test_mae,
                                        lake_id),
                                       index=[
                                           'train_rmse', 'test_rmse',
                                           'train_r2', 'test_r2',
                                           'train_mae', 'test_mae',
                                           'lake_id']
                                       ).T
            performance['Model'] = database
            performances.append(performance)
        except:
            print(lake_id, 'is empty')
    performances = pd.concat(performances)
    output_dir = os.path.join(root_dir, 'result', 'EAR5', 'exper2')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    performances.to_csv(os.path.join(output_dir, f'{database}.csv'), index=False)

    # exper3
    insitu_dir = os.path.join(root_dir, 'data', 'ccilakes')
    lake_list = pd.read_csv(os.path.join(insitu_dir, 'lake_info', 'ESA_CCI.csv'))
    obs_dir = os.path.join(insitu_dir, 'lswts')
    pred_dir = os.path.join(insitu_dir, 'mets')
    performances = []
    for i in trange(len(lake_list)):
        lake_id = lake_list.LakeID.values[i]
        try:
            # lake_id = 2
            met = pd.read_csv(os.path.join(pred_dir, f'met_{lake_id}.csv'))
            met['Date'] = pd.to_datetime(met['Date'], format='mixed')
            lswt = pd.read_csv(os.path.join(obs_dir, f'lake_id_{lake_id}.csv'))
            lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
            data_set = pd.merge(met, lswt, on='Date', how='inner')
            data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
            data_set['LSWT_pred'] = data_set['MixedTemp'].values
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set['Year'] = data_set['Date'].dt.year
            data_set['Mask'] = data_set['LSWT'].notnull().values.astype(int)
            data_set = data_set.query('Mask==1').reset_index(drop=True)
            #
            cal_set = data_set.query('Year >= 1995 and Year <= 2010').reset_index(drop=True)
            val_set = data_set.query('Year > 2010').reset_index(drop=True)
            #
            train_obs, train_pred = cal_set['LSWT'].values, cal_set['LSWT_pred'].values
            test_obs, test_pred = val_set['LSWT'].values, val_set['LSWT_pred'].values
            #
            train_rmse = RMSE(train_obs, train_pred)
            test_rmse = RMSE(test_obs, test_pred)
            #
            train_r2 = R_squared(train_obs, train_pred)
            test_r2 = R_squared(test_obs, test_pred)
            #
            train_mae = MAE(train_obs, train_pred)
            test_mae = MAE(test_obs, test_pred)
            # metrics calculation
            performance = pd.DataFrame((train_rmse, test_rmse,
                                        train_r2, test_r2,
                                        train_mae, test_mae,
                                        lake_id),
                                       index=[
                                           'train_rmse', 'test_rmse',
                                           'train_r2', 'test_r2',
                                           'train_mae', 'test_mae',
                                           'lake_id']
                                       ).T
            performance['Model'] = database
            performances.append(performance)
        except:
            print(lake_id, 'is empty')
    performances = pd.concat(performances)
    output_dir = os.path.join(root_dir, 'result', 'ERA5', 'exper3')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    performances.to_csv(os.path.join(output_dir, f'{database}.csv'), index=False)
