# modules
import torch
import numpy
import math
import logging

__all__ = [
    # EarlyStopping metrics
    'EarlyStopping',
    # logging metrics
    'FlushableStreamHandler',
    # evaluation metrics
    'IA', 'MAE', 'MARE', 'NRMSE', 'NRMSE_tensor_cal', 'RMSE', 'RMSE_tensor_cal', 'R_squared', 'R_squared_tensor_cal', 'SE',
    # physical operations
    'calculate_thermal_storage_change_rate', 'calculate_net_heat_fluxes',
    'calculate_heat_flux_Ice', 'calculate_heat_flux_latent', 'calculate_heat_flux_sensible',
    'calculate_wind_speed_10m','calculate_air_density',
    'calculate_vapour_pressure_air', 'calculate_vapour_pressure_saturated',
    ## mmd operations
    'rbf_kernel'
]

#
# Configure logging with immediate console flush for INFO and above levels
class FlushableStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        if record.levelno >= logging.INFO:
            self.flush()  # Ensure immediate console output flush for INFO level and above

###############################
# evaluation metrics
##################################
def R_squared(obs,pred):  # the same to NSE; Nash–Sutcliffe efficiency
    'ndarray -> ndarray'
    SSR=numpy.sum(numpy.square(pred-obs))
    SST=numpy.sum(numpy.square(numpy.average(obs)-obs))
    R_squared=(1 - SSR / SST)
    return R_squared
def RMSE(obs,pred):
    'ndarray -> ndarray'
    MSE=numpy.mean(numpy.square(obs-pred))
    RMSE=numpy.power(MSE,0.5)
    return RMSE
def NRMSE(obs,pred):
    'ndarray -> ndarray'
    MSE=numpy.mean(numpy.square(obs-pred))
    RMSE=numpy.power(MSE,0.5)
    scaler=numpy.max(obs)-numpy.min(obs)
    NRMSE=RMSE/scaler
    return NRMSE
def MAE(obs,pred):
    'ndarray -> ndarray'
    MAE=numpy.mean(numpy.abs(obs-pred))
    return MAE
def MARE(obs,pred):
    'ndarray -> ndarray'
    MARE=numpy.mean(numpy.abs((obs - pred)/obs))
    return MARE
def IA(obs,pred): # index of agreement; 0-1
    'ndarray -> ndarray'
    unexplained_error=numpy.sum(numpy.square(pred-obs))
    total_error=numpy.sum(numpy.square(numpy.abs(numpy.average(obs) - pred)+numpy.abs(numpy.average(obs)-obs)))
    IA =(1 - unexplained_error / total_error)
    return IA
def SE(obs,pred): # standard error
    'ndarray -> ndarray'
    mean_error=numpy.mean(numpy.abs(obs-pred))
    var=numpy.mean(numpy.square(pred-obs-mean_error))/(len(obs)-1)
    SE= numpy.power(var,0.5)
    return SE
def R_squared_tensor_cal(obs,pred):
    'tensor -> ndarray'
    SSR = torch.sum(torch.square(pred - obs))
    SST = torch.sum(torch.square(torch.mean(obs) - obs))
    R_squared=(1 - SSR / SST)
    return R_squared.cpu().detach().numpy()
def RMSE_tensor_cal(obs,pred):
    'tensor -> ndarray'
    MSE=torch.mean(torch.square(obs-pred))
    RMSE=torch.pow(MSE,0.5)
    return RMSE.cpu().detach().numpy()
def NRMSE_tensor_cal(obs,pred):
    'tensor -> ndarray'
    MSE=torch.mean(torch.square(obs-pred))
    RMSE=torch.pow(MSE,0.5)
    scaler=torch.max(obs)-torch.min(obs)
    NRMSE=RMSE/scaler
    return NRMSE.cpu().detach().numpy()

###############################
# physical operations
##################################

def calculate_thermal_storage_change_rate(T_s):
    """
    Args:
        T_s (seq_len): the pred temperature of the water surface layer (degC)

    Returns:
        lake surface thermal storage change rate at each timestep (W m-2)
         --> tensor (seq_len[1:-1])
    """
    # T_s = T_s_batch
    # Compute energy in surface layer for each timestep (result is in J)
    c = 4186  # specific heat of water (J/(kg degC))
    rho = 1000 # surface water density, kg m-3
    dz = .5  # assumed thickness of surface layer (m)
    dt = 86400 # time step 1 day, 86400s
    # Compute time derivative dT/dt using finite differences
    dTdz_top = torch.zeros_like(T_s)
    # Use central differences for interior points
    dTdz_top[1:-1] = (T_s[2:] - T_s[:-2]) / (2 * dt)
    # # Use forward/backward differences for boundaries
    dTdz_top[0] = (T_s[1] - T_s[0]) / dt
    dTdz_top[-1] = (T_s[-1] - T_s[-2]) / dt
    # lake surface thermal storage at each timestep, (W/m2)
    Q_t = dTdz_top * rho * dz * c
    return Q_t[1:-1]

def calculate_net_heat_fluxes(phy, T_s):
    """
    Calculate net heat flux components (F)

    Note:
        Ice-free/open-water calculations were modified from:
            Read et al. (2019). Water Resour. Res. 55, 9173–9190. https://doi.org/10.1029/2019WR024922
            Jia et al. (2019). SIAM Int. Conf. Data Mining, SDM 2019 558–566. https://doi.org/10.1137/1.9781611975673.63
            Willard et al. (2021). Water Resour. Res. 57, 7. https://doi.org/10.1029/2021WR029579
            Hipsey et al. (2019). Geoscientific Model Development. 12 (1), 473-523. https://doi.org/10.5194/gmd-12-473-2019
            #
            equations see Hipsey. et al
            source codes see Read. et al, Jia. et al, and Willard. et al
        Ice-cover equations were in:
            Wanders et al. (2019). Water Resour. Res. 55(4), 2760–2778. https://doi.org/10.1029/2018WR023250
            Rogers et al. (1995). Limnology and Oceanography. 40(2), 374–385. https://doi.org/10.4319/lo.1995.40.2.0374
        ##################################
        Details in all equations and their references can also be found in Supporting Information

    Args:
        phy (seq_len, n_vars): Physical variables:['SWdown', 'LWdown', 'AirTemp', 'RelHum', 'WindSpeed', 'Ice']
        T_s (seq_len): Temperatures of surface layer (deg_C)

    Returns:
        Net energy influx over each adjacent timestep (W m-2)
         --> tensor (seq_len[1:-1])
    """
    # phy = phy_batch
    # T_s = T_s_batch
    #-------------------------------------------Calculate radiation heat flux components (R: R_sw_in, R_lw_in, R_lw_emit)---------------------------------------------------------#
    e_s = 0.985 # emissivity of surface water, unitless
    alpha_sw = 0.07 # shortwave open-water albedo, unitless
    alpha_sw_si = 0.8 # shortwave snow-ice albedo, unitless
    alpha_lw = 0.03 # longwave open-water albedo, unitless
    sigma = 5.67e-8 # Stefan-Boltzmann constant, W m−2 K−4
    #  Use central differences to approximate radiation over time step
    R_sw = (phy[1:,0]+phy[:-1,0])/2 # downward shortwave radiation, W m−2
    R_lw = (phy[1:,1]+phy[:-1,1])/2 # downward longwave radiation, W m−2
    #
    R_sw_in = R_sw*(1-alpha_sw) # incoming shortwave radiation flux, W m−2
    R_lw_in = R_lw*(1-alpha_lw) # incoming longwave radiation flux, W m−2
    #
    R_lw_emit = e_s*sigma*(torch.pow(T_s+273.15, 4)) # Emitted longwave radiation flux, W m−2
    R_lw_emit = (R_lw_emit[1:]+R_lw_emit[:-1])/2
    #-------------------------------------------Calculate latent (E), sensible (H)----------------------------------------------------------#
    #  Use central differences to approximate radiation over time step
    t_s = T_s[:-1]
    t_s2 = T_s[1:]
    air_temp = phy[:-1,2] # air temperature at 2m height, deg_C
    air_temp2 = phy[1:,2]
    rel_hum = phy[:-1,3] # relative humidity at 2m height, 0-100%
    rel_hum2 = phy[1:,3]
    wind_speed = phy[:-1, 4] # wind speed at 10m height, m s-1
    wind_speed2 = phy[1:, 4]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, wind_speed)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, wind_speed)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, wind_speed2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, wind_speed2)
    E = (E + E2)/2
    H = (H + H2)/2
    #-------------------------------------------Calculate ice-cover penetrated shortwave radiation flux (R_sw_p)----------------------------------------------------------#
    R_sw_p= R_sw*(1-alpha_sw_si) # penetrated shortwave radiation flux, W m−2
    #-------------------------------------------Calculate ice conduct heat (H_ice)----------------------------------------------------------#
    H_ice = calculate_heat_flux_Ice(t_s)
    H_ice2 = calculate_heat_flux_Ice(t_s2)
    H_ice = (H_ice + H_ice2)/2
    #---------------------------------------------Net energy flux (F)-----------------------------------------------------------------------#
    ice = phy[1:-1, 5]
    F_ice_free = R_sw_in[:-1] + R_lw_in[:-1] - R_lw_emit[:-1] - H[:-1] - E[:-1]
    F_ice_cover = R_sw_p[:-1] - H_ice[:-1]
    #
    F = F_ice_free.clone()
    F[ice == 1] = F_ice_cover[ice == 1].clone()
    # the net energy flux over the same period i.e., time step
    return F

##################################
def calculate_air_density(air_temp, rh):
    # equations see Hipsey. et al
    # source codes see Read. et al, Jia. et al, and Willard. et al
    #
    # Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    # atmospheric pressure
    p = 1013.  # mb

    # water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)

    # water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure / (p - vapPressure)
    # air density (kg m-3)
    rho_a =(1.0 / c_gas * (1 + r) / (1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100
    return rho_a


def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    # equations see Hipsey. et al
    # source codes see Read. et al, Jia. et al, and Willard. et al
    #
    # calculate air density
    rho_a = calculate_air_density(air_temp, rel_hum)

    # specific heat capacity of air in J/(kg*C)
    c_a = 1005.

    # bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    # wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    # U_10 = wind_speed

    # sensible heat flux, W m−2
    H = -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)
    return H

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    # equations see Hipsey. et al
    # source codes see Read. et al, Jia. et al, and Willard. et al
    #
    # air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    # bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    # latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    # wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
    #
    # ratio of molecular weight of water to that of dry air
    omega = 0.622

    # air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    # latent heat flux, W m−2
    E = -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)
    return E

def calculate_vapour_pressure_air(rel_hum, temp):
    # equations see Hipsey. et al
    # source codes see Read. et al, Jia. et al, and Willard. et al
    #
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)


def calculate_vapour_pressure_saturated(temp):
    # Equations see Hipsey et al.
    # Returns in millibars
    # Use math.log(10) as a constant multiplier.
    exponent = (9.28603523 - (2332.37885 / (temp + 273.15))) * math.log(10)
    return torch.exp(exponent)

def calculate_wind_speed_10m(ws, ref_height=2.):
    # equations see Hipsey. et al
    # source codes see Read. et al, Jia. et al, and Willard. et al
    #
    c_z0 = 0.001  # default roughness
    return ws * (math.log(10.0 / c_z0) / math.log(ref_height / c_z0))

def calculate_heat_flux_Ice(T_s):
    # equations see Wanders. et al
    k_wi=0.57 # the heat conductivity between ice and water, W m−1 deg_C-1
    delta_wi=0.1 # an assumed thickness of ice-water interface with conduction, m
    H_ice = k_wi * (T_s/delta_wi) # Ice conductive heat flux, W m−2
    return H_ice


def rbf_kernel(X, n_kernels=5, mul_factor=2.0, bandwidth=None):
    """
    Compute a multi-scale RBF kernel matrix.
    source code see: https://github.com/ZongxianLee/MMD_Loss.Pytorch.git & https://github.com/yiftachbeer/mmd_loss_pytorch.git
    Args:
        X (Tensor): Input tensor of shape [n_samples, feature_dim].
        n_kernels (int): Number of Gaussian kernels to use. Default is 5.
        mul_factor (float): Multiplicative factor for scaling the bandwidth. Default is 2.0.
        bandwidth (float, optional): Fixed bandwidth. If None, compute adaptively.

    Returns:
        Tensor: Kernel matrix of shape [n_samples, n_samples].
    """

    bandwidth_multipliers = mul_factor ** (
        torch.arange(n_kernels, dtype=X.dtype, device=X.device) - n_kernels // 2
    )

    L2_distances = torch.cdist(X, X).pow(2)

    if bandwidth is None:
        n_samples = X.shape[0]
        bandwidth = L2_distances.sum() / (n_samples * (n_samples - 1))
        bandwidth = bandwidth.to(X.device)

    scaled_bandwidths = (bandwidth * bandwidth_multipliers).view(-1, 1, 1)
    kernels = torch.exp(-L2_distances.unsqueeze(0) / scaled_bandwidths)

    return kernels.sum(dim=0)

###############################
# model EarlyStopping
##################################
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    def __call__(self, val_loss):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0