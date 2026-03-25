import numpy as np
import pandas as pd
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Must be set before torch is imported
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from tqdm import trange, tqdm
import random
import multiprocessing as mp

from model.Module.Dataset import *
from model.Module.Models import *
from model.Module.Utils import *

__all__ = [
    "seed_worker",
    "init_model_weights",
    "save_model",
    "load_model",
    "train",
    "predict",
    "project",
    'cali_task',
    "cali_mp",
    "domain_adaptation"
]


###############################
# reproducible
##################################
def seed_worker(seed=100):
    # Set Python built-in random seed
    random.seed(seed)
    # Set NumPy seed
    np.random.seed(seed)
    # Set PyTorch CPU seed
    torch.manual_seed(seed)
    try:
        # Enable deterministic algorithms (if supported)
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("Unable to enable deterministic algorithms:", e)
    # Create and seed a torch Generator
    g = torch.Generator()
    g.manual_seed(seed)

    # If CUDA is available, set its seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
        except Exception as e:
            print("Unable to set cudnn deterministic mode:", e)
        torch.backends.cudnn.benchmark = False

    # If MPS (Apple Silicon) is available, set its seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        try:
            torch.backends.mps.deterministic = True
        except Exception as e:
            print("Unable to set MPS deterministic mode:", e)
        torch.backends.mps.benchmark = False


###############################
# xavier_uniform and zeros model weight initial
##################################
def init_model_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight.data)
        torch.nn.init.zeros_(module.bias.data)
        # print('linear layer initialization done')
    if isinstance(module, torch.nn.LSTM):
        for param in module._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(module._parameters[param])
        # print('LSTM layer initialization done')
    if isinstance(module, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias.data is not None:
            torch.nn.init.zeros_(module.bias.data)
        # print('Conv1d layer initialization done')
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias.data is not None:
            torch.nn.init.zeros_(module.bias.data)
        # print('Conv2d layer initialization done')


###############################
# model weight save
##################################
def save_model(model, optimizer, save_path, weight_only=False):
    if weight_only:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        state = {'state_dict': model_state, 'optimizer': optimizer_state}
        torch.save(state, save_path)
    else:
        torch.save(model, save_path)


def load_model(model_dir):
    try:
        pretrain_model = torch.load(
            model_dir,
            map_location='cpu',
            weights_only=False
        )
        # pretrain_model_struc = pd.read_csv(
        #     os.path.join(model_dir, 'cal_result_pre_train.csv')
        # )
        print('PRE-TRAINER LOAD SUCCESS')
        ###--------------------------------console print only--------------------------------###
    except Exception as e:

        pretrain_model = None
        # pretrain_model_struc = None
        print(e)
        print('NO PRE-TRAINER LOADED !!')
    return pretrain_model

###############################
# model mini-batch train
##################################
def train(mdl, optimizer, hyper_parameter, train_loader, Configs, verbose=0):
    losses_train = []
    # Set seed for reproducibility
    seed = Configs['seed']
    seed_worker(seed)
    # unpack configs
    device = Configs['device']
    save_point = hyper_parameter['Epoch']
    # Determine whether verbose epochs
    num_epochs = hyper_parameter['Epoch']
    epochs = range(num_epochs)
    if verbose > 0:
        epochs = trange(num_epochs)
    # Move model and loss function to device
    mdl = mdl.to(device)
    loss_func = HybridLoss()
    loss_func = loss_func.to(device)
    ## ----------------------------model train ---------------------------- ##
    for epoch in epochs:
        train_loss = []
        train_loss_data = []
        train_loss_ec = []
        mdl.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_i, data in enumerate(train_loader):
            # get the inputs -- a list of [inputs, labels]
            x, y, phys = [X.to(device) for X in data]
            optimizer.zero_grad()
            y_pred, _ = mdl(x)
            # state = (State[0].data,State[1].data)
            loss, loss_data, loss_ec = loss_func(
                y_pred, y, phys,
                hyper_parameter['lam_ec'],
                Configs['physics_informed']
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            train_loss.append(loss.detach().cpu().item())
            # print(loss_ec.detach().cpu().item())
            # print(train_loss/N)
            # print(f"batch {batch_i}th with batch size of {train_x.shape[0]}")
        train_loss = epoch_loss / num_batches if num_batches > 0 else np.nan
        losses_train.append(train_loss)
        if (epoch + 1) % 50 == 0 and verbose > 1:
            print(f'---epoch{epoch + 1}----train loss: {round(train_loss, 5)}')
        ## ----------------------------model val ---------------------------- ##
        # val_loss = []
        # mdl.eval() # prep model for evaluation
        # for batch_i, data in enumerate(val_loader):
        #     x, y, _, masks = [X.to(device) for X in data]
        #     y_pred,_ = mdl(x)
        #     loss,_,_ = loss_func(y_pred, y, _, masks, _, _, False)
        #     val_loss.append(loss.detach().cpu().item())
        # val_loss = np.asarray(val_loss).mean()
        # if (epoch + 1) % 50 == 0:
        #     print(f'---epoch{epoch + 1}---val loss: {val_loss}----train loss: {train_loss}')
        # early_stopping(val_loss)
        # stop = early_stopping.early_stop
        # if early_stop and stop:
        #     save_point = epoch
        #     print('\r', "early stop at",save_point, end='...')
        #     break
    return mdl, optimizer, losses_train, save_point


###############################
# model predict
##################################

def predict(seq_length, data_pack, mdl, seed, device):
    """
        Safely obtain model predictions and performance metrics.

        If any error occurs during the prediction or performance computation,
        the function prints the error message and returns pred as None and performance
        as an empty DataFrame with index ['r_squared', 'rmse', 'nrmse', 'mae', 'mare'].

        Args:
            seq_length (int): Length of the sliding window sequence.
            data_pack (list): List containing test data ([features,labels,phys,Masks]).
            mdl (torch.nn.Module): The model used for prediction.
            seed (int): Seed for reproducibility.
            device (torch.device): The device to run computations on.

        Returns:
            tuple: A tuple (pred, performance) where:
                - pred (np.ndarray or None): Model predictions with shape (batch_size * seq_len).
                - performance (pd.DataFrame): Performance metrics DataFrame with index:
                  ['r_squared', 'rmse', 'nrmse', 'mae', 'mare'].
        """
    # Set seed for reproducibility in worker processes.
    seed_worker(seed)

    # Create sliding windows for features and labels.
    data = [
        sliding_windows(data_pack[0], seq_length, seq_length, True),  # features
        sliding_windows(data_pack[1], seq_length, seq_length, True)  # labels
    ]
    # Move to device and get predictions
    mdl.to(device)
    mdl.eval()
    with torch.no_grad():
        pred = mdl(data[0].to(device))[0].cpu().numpy().flatten()
        obs = data[1].cpu().numpy().flatten()

    # filter invalid values (NaN or non-positive)
    valid_mask = ~np.isnan(obs) & (obs > 0)
    obs_valid, pred_valid = obs[valid_mask], pred[valid_mask]

    # Check for empty valid arrays
    if obs_valid.size == 0 or pred_valid.size == 0:
        print("No valid observations after masking and filtering!")
        empty_performance = pd.DataFrame(
            index=['r2', 'rmse', 'nrmse', 'mae', 'mare']
        ).T
        empty_generation = pd.DataFrame(
            index=['Observed', 'Predicted', 'Error']
        ).T
        return empty_generation, empty_performance
    else:
        generation = pd.DataFrame(np.stack((obs_valid, pred_valid), axis=1),
                                  columns=['Observed', 'Predicted'])
        generation['Error'] = (pred_valid.astype(float) - obs_valid.astype(float)) / (
                abs(obs_valid.astype(float)) + abs(pred_valid.astype(float)))

        # Helper function: safely calculate a metric.
        def safe_calc(func, *args, **kwargs):
            try:
                if len(args[0]) == 0 or len(args[1]) == 0:
                    return np.nan
                return func(*args, **kwargs)
            except Exception as e:
                return np.nan

        # Compute performance metrics while keeping the same DataFrame format.
        performance = pd.DataFrame(
            [
                safe_calc(R_squared, obs_valid, pred_valid),
                safe_calc(RMSE, obs_valid, pred_valid),
                safe_calc(NRMSE, obs_valid, pred_valid),
                safe_calc(MAE, obs_valid, pred_valid),
                safe_calc(MARE, obs_valid, pred_valid),
            ],
            index=['r2', 'rmse', 'nrmse', 'mae', 'mare']
        ).T
        return generation, performance


###############################
# model project
##################################

def project(seq_length, data_pack, mdl, seed, device):
    # Set seed for reproducibility in worker processes.
    seed_worker(seed)

    # Create sliding windows for features and labels.
    data = [
        sliding_windows(data_pack[0], seq_length, seq_length, True),  # features
        sliding_windows(data_pack[1], seq_length, seq_length, True)  # mask
    ]
    # Move to device and get predictions
    data = [set.to(device) for set in data]
    mdl.to(device)
    mdl.eval()
    with torch.no_grad():
        pred = mdl(data[0].to(device))[0].cpu().numpy().flatten()
        mask = data[1].cpu().detach().numpy().flatten()
    #
    pred = pred[mask == 1]

    return pred





###############################
# model calibration
##################################
def cali_task(task):
    # task = tasks[0]
    (model_idx, hyper_parameter, Configs, train_data_pack, test_data_pack, pretrain_model,
     device, seed, verbose) = task
    # unpack
    seed_worker(seed)
    torch_g = torch.Generator().manual_seed(seed)
    # Precompute the sliding window transforms for the training data.
    # [features,labels,phys,Masks] = train_data_pack
    train_data = [sliding_windows(sets, Configs['seq_len'], Configs['shift_size'], Configs['padding'])
                  for sets in train_data_pack]
    train_data = [sets.to(device) for sets in train_data]
    # Create DataLoader for training using the precomputed ground and distill data.
    train_loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=hyper_parameter['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch_g
    )

    # Initialize the model: either load weights from pretrain_model or initialize from xavier_uniform_.
    if pretrain_model is not None:
        pretrain_model = pretrain_model.to(device)
        lstm_weight = pretrain_model.lstm_layers.state_dict()
        fc_weight = pretrain_model.fc.state_dict()
        hyper_parameter['hidden_dim'] = pretrain_model.hidden_dim_lstm
        hyper_parameter['layer_num'] = pretrain_model.layer_num_lstm
        hyper_parameter['hidden_dim_fc1'] = pretrain_model.hidden_dim_fc1
        hyper_parameter['hidden_dim_fc2'] = pretrain_model.hidden_dim_fc2
        mdl = MyLSTM(
            input_dim=Configs['input_dim'],
            output_dim=Configs['output_dim'],
            hidden_dim=hyper_parameter['hidden_dim'],
            layer_num=hyper_parameter['layer_num'],
            hidden_dim_fc1=hyper_parameter['hidden_dim_fc1'],
            hidden_dim_fc2=hyper_parameter['hidden_dim_fc2'],
            bidirecion=False,
            dropout_lstm=0,
            dropout_fc=0
        )
        mdl.lstm_layers.load_state_dict(lstm_weight)
        mdl.fc.load_state_dict(fc_weight)
        if Configs.get('freeze', False):
            # Freeze LSTM weights by setting requires_grad to False
            for param in mdl.lstm_layers.parameters():
                param.requires_grad = False
    else:
        mdl = MyLSTM(
            input_dim=Configs['input_dim'],
            output_dim=Configs['output_dim'],
            hidden_dim=hyper_parameter['hidden_dim'],
            layer_num=hyper_parameter['layer_num'],
            hidden_dim_fc1=hyper_parameter['hidden_dim_fc1'],
            hidden_dim_fc2=hyper_parameter['hidden_dim_fc2'],
            bidirecion=False,
            dropout_lstm=0,
            dropout_fc=0
        )
        mdl.apply(init_model_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, mdl.parameters()),
        lr=hyper_parameter['learning_rate']
    )
    # Train the model.
    mdl_trained, _, _, save_point = train(mdl, optimizer, hyper_parameter, train_loader, Configs, verbose)

    # Evaluate the trained model on training and testing data.
    seq_length = Configs['seq_len']
    train_generations, train_perform = predict(seq_length, train_data_pack, mdl_trained, seed, device)
    train_perform.columns = train_perform.columns.map(lambda x: 'train_' + x)
    test_generations, test_perform = predict(seq_length, test_data_pack, mdl_trained, seed, device)
    test_perform.columns = test_perform.columns.map(lambda x: 'test_' + x)

    param = pd.DataFrame.from_dict(hyper_parameter, orient='index').T
    result = pd.concat((train_perform, test_perform, param), axis=1)
    result['model_idx'] = model_idx
    train_generations['phase'] = 'train'
    test_generations['phase'] = 'test'
    generations = pd.concat((train_generations, test_generations), axis=0)
    generations['model_idx'] = model_idx
    # If the model is on the GPU, move it to CPU before returning
    # This if for parallel running on GPU
    if hasattr(mdl_trained, 'cpu'):
        mdl_trained = mdl_trained.cpu()
    return (result, mdl_trained, generations)


###############################
# model calibration in parallel
##################################
def cali_mp(Configs, Search_space, train_data_pack, test_data_pack,
            pretrain_model=None):
    """
    Parallelises the hyperparameter search over the provided Search_space.

    Args:
        Configs: Dictionary of configuration parameters.
        Search_space: Dictionary where each key corresponds to a hyperparameter
                      and its value is a list of values to search over.
        train_data_pack: Training data tuple (e.g., [features,labels,phys,Masks]).
        test_data_pack: Testing data pack.
        pretrain_model: Optional pretrained model.
        model_return: If True, return the list of trained models along with the results.
    """

    # Configs, Search_space = Configs_updated, Search_space_updated
    import itertools
    from scipy.stats.qmc import LatinHypercube, scale
    parallel = Configs.get('parallel', False)
    num_of_process = Configs.get('num_of_process', 0)
    device = Configs.get('device', 'cpu')
    verbose = Configs.get('verbose', 2)
    seed = Configs.get('seed', 100)
    sampler = Configs.get('sampler', 'LHC')
    # Separate hyperparameters into continuous (range) and fixed (discrete)
    continuous_params = {}
    fixed_params = {}
    for key, value in Search_space.items():
        # Check if the value is a list or tuple of two numbers (indicating a continuous range)
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            continuous_params[key] = value
        else:
            fixed_params[key] = value
    # Get continuous parameter names and determine the sampling dimension
    param_keys = list(continuous_params.keys())
    dim = len(param_keys)
    ### LatinHypercube optimse
    if sampler == 'LHC':
        num_params = Configs.get('num_params', 100)
        # Extract the lower and upper bounds for each hyperparameter.
        l_bounds = []
        u_bounds = []
        for key in param_keys:
            lower, upper = Search_space[key]
            l_bounds.append(lower)
            u_bounds.append(upper)
        # Create a LatinHypercube sampler for a space of dimension 'dim' with the specified seed
        lhs_sampler = LatinHypercube(d=dim, seed=seed)
        # Generate an array of shape (n_samples, dim) with values in [0, 1)
        samples = lhs_sampler.random(n=num_params)
        # Scale the samples from the unit hypercube to the specified parameter bounds.
        samples_scaled = scale(samples, l_bounds=l_bounds, u_bounds=u_bounds)
        # Create tasks with the scaled hyperparameter values.
        tasks = []
        model_idx = 0
        for i in range(num_params):
            hyper_parameter = {}
            for j, key in enumerate(param_keys):
                hyper_parameter[key] = samples_scaled[i, j]
            hyper_parameter.update(fixed_params)
            hyper_parameter = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in
                               hyper_parameter.items()}
            # print(hyper_parameter)
            tasks.append((model_idx, hyper_parameter, Configs, train_data_pack, test_data_pack, pretrain_model,
                          device, seed, verbose))
            model_idx += 1

    elif sampler == 'Grid_log':
        ### grid search
        # For each hyperparameter, create an array of num_params equally spaced values between its bounds.
        num_params = Configs.get('num_params', 5)
        parameter_values = {}
        for key in param_keys:
            lower, upper = Search_space[key]
            # Generate num_params equally spaced values in the interval [lower, upper]
            log_values = np.linspace(np.log(lower), np.log(upper), num_params)
            parameter_values[key] = np.exp(log_values)
        # Create the Cartesian product of all parameter values.
        # Each element in 'all_combinations' is a tuple containing one sample for each hyperparameter.
        all_combinations = list(itertools.product(*(parameter_values[key] for key in param_keys)))
        # Create tasks using each hyperparameter combination from the Cartesian product.
        tasks = []
        model_idx = 0
        for combination in all_combinations:
            hyper_parameter = dict(zip(param_keys, combination))
            hyper_parameter.update(fixed_params)
            hyper_parameter = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in
                               hyper_parameter.items()}
            # print(hyper_parameter)
            tasks.append((model_idx, hyper_parameter, Configs, train_data_pack, test_data_pack, pretrain_model,
                          device, seed, verbose))
            model_idx += 1


    elif sampler == 'Grid':
        ### grid search
        # For each hyperparameter, create an array of num_params equally spaced values between its bounds.
        num_params = Configs.get('num_params', 5)
        parameter_values = {}
        for key in param_keys:
            lower, upper = Search_space[key]
            # Generate num_params equally spaced values in the interval [lower, upper]
            parameter_values[key] = np.linspace(lower, upper, num_params)
        # Create the Cartesian product of all parameter values.
        # Each element in 'all_combinations' is a tuple containing one sample for each hyperparameter.
        all_combinations = list(itertools.product(*(parameter_values[key] for key in param_keys)))
        # Create tasks using each hyperparameter combination from the Cartesian product.
        tasks = []
        model_idx = 0
        for combination in all_combinations:
            hyper_parameter = dict(zip(param_keys, combination))
            hyper_parameter.update(fixed_params)
            hyper_parameter = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in
                               hyper_parameter.items()}
            # print(hyper_parameter)
            tasks.append((model_idx, hyper_parameter, Configs, train_data_pack, test_data_pack, pretrain_model,
                          device, seed, verbose))
            model_idx += 1
    else:
        tasks = []
        model_idx = 0
        for Hyper_param_comb_list in itertools.product(*Search_space.values()):
            hyper_parameter = dict(zip(Search_space.keys(), Hyper_param_comb_list))
            tasks.append((model_idx, hyper_parameter, Configs, train_data_pack, test_data_pack, pretrain_model,
                          device, seed, verbose))
            model_idx += 1

    if parallel:
        try:
            # Run all tasks in parallel.
            with mp.Pool(processes=num_of_process) as pool:
                # results_list = pool.map(cali_task, tasks)
                results_list = list(tqdm(pool.imap(cali_task, tasks), total=len(tasks),
                                         desc=f"Calibration Progress in parallel on {device}"))
        except Exception as e:
            print("Error occurred: Run all tasks in parallel.")
            print("Reason:", e)
            print("Roll back to serial execution.")
            # Run all tasks in serial.
            results_list = []
            for task in tqdm(tasks, desc=f"Calibration Progress in serial on {device}"):
                results_list.append(cali_task(task))
    else:
        # Run all tasks in serial.
        results_list = []
        for task in tqdm(tasks, desc=f"Calibration Progress in serial on {device}"):
            results_list.append(cali_task(task))
    # Collect results and (optionally) models.
    results = [r[0] for r in results_list]
    model_list = [r[1] for r in results_list]
    generation_list = [r[2] for r in results_list]

    results = pd.concat(results).reset_index(drop=True)
    generations = pd.concat(generation_list).reset_index(drop=True)
    return results, model_list, generations


###############################
#  mini-batch domain adaption
##################################
def domain_adaptation(Configs, Search_space, source_data_pack, mid_data_pack, pretrain_model):
    import itertools
    device = get_device(Configs.get('use_gpu', False))
    verbose = Configs.get('verbose', 2)
    # Set seed for reproducibility
    seed = Configs.get('seed', 100)
    seed_worker(seed)
    torch_g = torch.Generator().manual_seed(seed)
    #
    model_idx = 0
    for Hyper_param_comb_list in itertools.product(*Search_space.values()):
        hyper_parameter = dict(zip(Search_space.keys(), Hyper_param_comb_list))
        source_data = [sliding_windows(sets, Configs['seq_len'], Configs['shift_size'], Configs['padding'])
                      for sets in source_data_pack]
        mid_data = [sliding_windows(sets, Configs['seq_len'], Configs['shift_size'], Configs['padding'])
                      for sets in mid_data_pack]
        source_data = [sets.to(device) for sets in source_data]
        mid_data = [sets.to(device) for sets in mid_data]
        # Create DataLoader for training using the precomputed ground and distill data.
        source_loader = DataLoader(
            TensorDataset(*source_data),
            batch_size=hyper_parameter['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=torch_g
        )
        mid_loader = DataLoader(
            TensorDataset(*mid_data),
            batch_size=hyper_parameter['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=torch_g
        )
        #
        losses_train = []
        save_point = hyper_parameter['Epoch']
        # Determine whether verbose epochs
        num_epochs = hyper_parameter['Epoch']
        epochs = range(num_epochs)
        if verbose > 0:
            epochs = trange(num_epochs)
        # Move model and loss function to device
        lstm_weight = pretrain_model.lstm_layers.state_dict()
        fc_weight = pretrain_model.fc.state_dict()
        hyper_parameter['hidden_dim'] = pretrain_model.hidden_dim_lstm
        hyper_parameter['layer_num'] = pretrain_model.layer_num_lstm
        hyper_parameter['hidden_dim_fc1'] = pretrain_model.hidden_dim_fc1
        hyper_parameter['hidden_dim_fc2'] = pretrain_model.hidden_dim_fc2
        mdl = MyLSTM(
            input_dim=Configs['input_dim'],
            output_dim=Configs['output_dim'],
            hidden_dim=hyper_parameter['hidden_dim'],
            layer_num=hyper_parameter['layer_num'],
            hidden_dim_fc1=hyper_parameter['hidden_dim_fc1'],
            hidden_dim_fc2=hyper_parameter['hidden_dim_fc2'],
            bidirecion=False,
            dropout_lstm=0,
            dropout_fc=0
        )
        mdl.lstm_layers.load_state_dict(lstm_weight)
        mdl.fc.load_state_dict(fc_weight)
        # if Configs.get('freeze', False):
        #     # Freeze LSTM weights by setting requires_grad to False
        #     for param in mdl.lstm_layers.parameters():
        #         param.requires_grad = False
        mdl = mdl.to(device)
        loss_func = Domain_adaptaion_Loss()
        loss_func = loss_func.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, mdl.parameters()),
            lr=hyper_parameter['learning_rate']
        )
        ## ----------------------------model train ---------------------------- ##
        for epoch in epochs:
            mdl.train()
            source_iter = iter(source_loader)
            mid_iter = iter(mid_loader)
            epoch_loss = 0.0
            num_batches = 0

            while True:
                try:
                    src_batch, src_labels, _ = next(source_iter)
                except StopIteration:
                    break
                try:
                    mid_batch, mid_labels, _  = next(mid_iter)
                except StopIteration:
                    mid_iter = iter(mid_loader)
                    mid_batch, mid_labels, _  = next(mid_iter)

                # Move data to device if necessary
                src_batch = src_batch.to(device)
                src_labels = src_labels.to(device)
                mid_batch = mid_batch.to(device)
                mid_labels = mid_labels.to(device)

                optimizer.zero_grad()
                src_pred, src_features = mdl(src_batch)
                mid_pred, mid_features = mdl(mid_batch)

                # Calculate the hybrid loss (e.g., MSE losses + MMD loss)
                loss, _, _ , _ = loss_func(
                    src_pred, src_labels, src_features,
                    mid_pred, mid_labels, mid_features,
                    1,
                    .1,
                    0.01
                )
                # print(loss.item())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            # Compute average loss for the epoch
            train_loss = epoch_loss / num_batches if num_batches > 0 else np.nan
            losses_train.append(train_loss)
            if (epoch + 1) % 10 == 0 and verbose > 1:
                print(f'--- Epoch {epoch + 1} --- Train Loss: {round(train_loss, 5)}')

        return mdl, optimizer, losses_train, save_point