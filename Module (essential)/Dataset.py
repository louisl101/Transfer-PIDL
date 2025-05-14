# Data processing module
import math
import numpy as np
import pandas as pd
import torch
import random
import os
#
__all__ = [
    "sliding_windows",
    "data_split",
    "get_pretrain_data",
    "get_synthetic_label"
]

def sliding_windows(data, seq_length, shift_size, padding):
    data_reconstructed=[]
    if data.ndim<2: data=np.expand_dims(data,axis=1)
    sliding_times=math.floor((data.shape[0] - seq_length) / shift_size)
    sliding_end_row=(sliding_times * shift_size) + seq_length
    # pad for one more extra sliding
    if data.shape[0]==sliding_end_row: #stands for exactly to the end, no padding needed
        pad_rows=0
    else:
        pad_rows = shift_size - (data.shape[0] - sliding_end_row)
    pad_data=np.zeros((pad_rows,data.shape[1]))
    data_padded=np.concatenate((data,pad_data))
    sliding_times_padded = math.floor((data_padded.shape[0] - seq_length) / shift_size)
    if sliding_times_padded < 0:
        sliding_times_padded = 0
    for j in range(data_padded.shape[1]):
        jth_data = []
        data_trunc=data_padded[:,j]
        for i in range(sliding_times_padded +1): ## data_reconstructed has (sliding time+1) rows
            _data = data_trunc[i*shift_size:(i*shift_size)+seq_length]
            jth_data.append(_data)
        jth_data = np.stack(jth_data,axis=0)
        data_reconstructed.append(jth_data)
    data_recons=np.stack(data_reconstructed,axis=-1)
    if padding:
        return torch.as_tensor(data_recons,dtype=torch.float32)
    else:
        return torch.as_tensor(data_recons[:-1],dtype=torch.float32)

def data_split(data_file,test_size=0.25,seed=100,shuffle=True):
    assert isinstance(data_file,  (pd.DataFrame,np.ndarray,torch.Tensor)), 'must be ndarray-like -> ndarray-like'
    # data random split
    random.seed(seed)
    if shuffle is False:
        train_indices = list(range(round(data_file.shape[0] * (1 - test_size))))
        test_indices = list(set(range(data_file.shape[0])).difference(set(train_indices)))
    else:
        test_indices = random.sample(range(data_file.shape[0]),round(data_file.shape[0] * test_size))
        test_indices.sort()
        # train_indices = [i for i in range(reconstructed_data.shape[0]) if i not in test_indices]
        train_indices = list(set(range(data_file.shape[0])).difference(set(test_indices)))
    try:
        test_set = data_file[test_indices]
        train_set = data_file[train_indices]
    except:
        test_set = data_file.loc[test_indices].reset_index(drop=True)
        train_set = data_file.loc[train_indices].reset_index(drop=True)
    return train_set,test_set


def get_pretrain_data(source_list, data_dir, exper, pretrain_scale=-1, iter_seed=0):
    """
    Parameters
    ----------
    source_list : list
        A list contains source lake IDs.
    data_dir : str
        Directory path containing data files.
    exper : str, optional
        Experiment type ('exper1', 'exper2', or 'exper3'), by default 'exper1'.
    pretrain_request : bool, optional
        Flag to indicate if return data for pretraining, by default False.
    domain_adapt_request : bool, optional
        Flag to indicate if return data for domain adaptation training, by default False.
    pretrain_scale : int, optional
        Number of lakes for pretraining, by default -1.
    (<0 for all, 0-1 for percentage, >=1 absolute number)
    iter_seed : int, optional
        Seed for internal random number generator, by default 0.
    """
    phys_var = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Ice']
    input_var = ['DOY', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed']
    train_sets = []  # No separate test data required if pretraining
    # Randomly sample lake IDs with fixed seed for reproducibility
    # (<0 for all, 0-1 for percentage, >=1 absolute number)
    lake_list = pd.DataFrame(source_list, columns=['LakeID'])
    if pretrain_scale < 0:
        lakeIDs = lake_list['LakeID'].sort_values().tolist()
    elif 0 < pretrain_scale < 1:
        sample_size = max(1, int(len(lake_list) * pretrain_scale))  # at least 1
        lakeIDs = lake_list.sample(n=sample_size, random_state=iter_seed)['LakeID'].sort_values().tolist()
    else:
        lakeIDs = lake_list.LakeID.sample(n=pretrain_scale, random_state=iter_seed).sort_values().tolist()

    for lake_id in lakeIDs:
        met = pd.read_csv(os.path.join(data_dir, 'mets', f'met_{lake_id}.csv'))
        met['Date'] = pd.to_datetime(met['Date'], format='mixed')
        lswt = pd.read_csv(os.path.join(data_dir, 'lswts', f'lake_id_{lake_id}.csv'))
        lswt['Date'] = pd.to_datetime(lswt['Date'], format='mixed')
        data_set = pd.merge(met, lswt, on='Date', how='inner')
        data_set['LSWT'] = data_set['lake_surface_water_temperature'].values - 273.15
        data_set['Year'] = data_set['Date'].dt.year
        if exper in ['exper1', 'exper2']:
            train_set = data_set.query('Year >= 1995 and Year <= 2020').reset_index(drop=True)
        else:
            train_set = data_set.query('Year >= 1995 and Year <= 2010').reset_index(drop=True)

        train_sets.append(train_set)

    train_sets = pd.concat(train_sets).reset_index(drop=True)

    # Package train and test data (features, labels, phys_vars)
    train_data_pack = [train_sets[input_var].values,
                       train_sets['LSWT'].values,
                       train_sets[phys_var].values
                       ]
    return [train_data_pack, train_data_pack]


def get_synthetic_label(synthetic_data_dir):
    import xarray as xr
    # Read ensemble dataset and split it
    synthetic_data = os.path.join(synthetic_data_dir, "output", "glm_output.nc")
    with xr.open_dataset(synthetic_data) as Temps:
        # Filter out spinup days: remove the first 2 years of data
        start_date = pd.to_datetime(Temps["time"].values[0])
        spinup_end_date = start_date + pd.DateOffset(years=2)
        Temps = Temps.sel(time=slice(spinup_end_date, None))
        synthetic_labels = Temps.sel(
            model=1,
            member=1,
            lat=Temps.lat.item(),
            lon=Temps.lon.item(),
            z=0
        )["temp"]
        obs_labels = Temps.sel(
            model=2,
            member=1,
            lat=Temps.lat.item(),
            lon=Temps.lon.item(),
            z=0
        )["temp"]
        synthetic_labels_df = pd.DataFrame({
            "Date": pd.to_datetime(synthetic_labels.time.values),  #
            "LSWT_sims": synthetic_labels.values,
            "LSWT_obs": obs_labels.values
        })
    return synthetic_labels_df