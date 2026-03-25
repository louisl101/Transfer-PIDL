# ------------------------------ Load Packages ------------------------------
library(glmtools)
library(GLM3r)
library(lutz)
library(lubridate)
library(tidyverse)
library(sf)
library(xml2)
library(jsonlite)
library(LakeEnsemblR)
library(future)
library(future.apply)
options(scipen = 999)
set.seed(100)
# ------------------------------ Function Definitions ------------------------------
# Function to detect the script directory based on command line arguments
getScriptDir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fileArg <- grep("^--file=", args, value = TRUE)
  if (length(fileArg) > 0) {
    scriptPath <- normalizePath(sub("^--file=", "", fileArg))
    return(dirname(scriptPath))
  } else {
    stop("Unable to determine script directory. Are you running this using Rscript?")
  }
}

# Generate a hypsographic curve with specified spacing
generate_hypsographic_curve <- function(Z_max, surface_area, interval = 4) {
  if (Z_max > interval) {
    depths <- seq(0, Z_max, by = interval)
  } else {
    depths <- seq(0, Z_max, length.out = 4)
  }
  surface_areas <- surface_area * (1 - (depths / Z_max))^2
  hypsographic_curve <- data.frame(Depth_meter = round(depths, 1),
                                   Area_meterSquared = round(surface_areas, 1))
  return(hypsographic_curve)
}

# Generate standard inputs for LakeEnsemblR
generate_LER_standard <- function(lake_infor, meteo_file, wtemp_lswt) {
  lake_id <- lake_infor$LakeID
  tr <- lake_infor$TR
  surface_area <- lake_infor$Lake_area * 1e6     # km2 to m2
  Vol_total <- lake_infor$Vol_total * 1e-3 * 1e9   # mcm to km3 to m3
  Z_max <- (3 * Vol_total) / surface_area          # assuming a conical shape
  bathymetry_standard <- generate_hypsographic_curve(Z_max, surface_area, interval = 4)

  # Process lake surface water temperature data
  wtemp_lswt <- wtemp_lswt %>% filter(!is.na(lake_surface_water_temperature))
  wtemp_lswt$Water_Temperature_celsius <- wtemp_lswt$lake_surface_water_temperature - 273.15
  wtemp_lswt$Depth_meter <- 0
  wtemp_lswt$datetime <- ymd(wtemp_lswt$Date)
  wtemp_profile_standard <- wtemp_lswt %>% select(datetime, Depth_meter, Water_Temperature_celsius)

  # Initial temperature profile guess based on lake type
  initial_temp_list <- list(
    NF = 3.3,
    NC = 6.1,
    NT = 9.8,
    NW = 16.3,
    NH = 22.1,
    TH = 29.0,
    SH = 24.2,
    SW = 15.7,
    ST = 7.9
  )
  initial_temp <- initial_temp_list[[tr]]
  initial_standard <- data.frame(Depth_meter = bathymetry_standard$Depth_meter,
                                 Water_Temperature_celsius = initial_temp)

  # Process meteorological data
  meteo_file$datetime <- ymd(meteo_file$Date)
  meteo_file <- meteo_file %>%
    rename(
      Ten_Meter_Elevation_Wind_Speed_meterPerSecond = WindSpeed,
      Air_Temperature_celsius = AirTemp,
      Relative_Humidity_percent = RelHum,
      Shortwave_Radiation_Downwelling_wattPerMeterSquared = ShortWave,
      Longwave_Radiation_Downwelling_wattPerMeterSquared = LongWave,
      Precipitation_millimeterPerDay = Rain,
      Snowfall_millimeterPerDay = Snow
    )
  column_order <- c(
    "datetime",
    "Ten_Meter_Elevation_Wind_Speed_meterPerSecond",
    "Air_Temperature_celsius",
    "Relative_Humidity_percent",
    "Shortwave_Radiation_Downwelling_wattPerMeterSquared",
    "Longwave_Radiation_Downwelling_wattPerMeterSquared",
    "Precipitation_millimeterPerDay",
    "Snowfall_millimeterPerDay"
  )
  meteo_standard <- meteo_file %>% select(all_of(column_order))

  return(list(bathymetry_standard, wtemp_profile_standard, initial_standard, meteo_standard))
}

# Write configuration updates to the YAML configuration file
write_configs <- function(config_file, lake_infor, start, end, best_p = NULL) {
  lake_id <- lake_infor$LakeID
  lat <- lake_infor$Lat_centre
  lon <- lake_infor$Lon_centre
  surface_area <- lake_infor$Lake_area * 1e6
  Vol_total <- lake_infor$Vol_total * 1e-3 * 1e9
  Z_max <- (3 * Vol_total) / surface_area
  surface_elev <- lake_infor$Elevation
  timezone <- tz_lookup_coords(lat, lon, method = "accurate")
  local_time <- now(tzone = timezone)
  offset_hours <- as.numeric(format(local_time, "%z")) / 100

  ### Update configuration with optimal parameters for GLM
  input_yaml_multiple(file = config_file, value = paste0("lake_id_", lake_id), key1 = "location", key2 = "name")
  input_yaml_multiple(file = config_file, value = paste0(start, " 00:00:00"), key1 = "time", key2 = "start")
  input_yaml_multiple(file = config_file, value = paste0(end, " 00:00:00"), key1 = "time", key2 = "stop")

  ### Update configuration with lake-specific information
  input_yaml_multiple(file = config_file, value = lat, key1 = "location", key2 = "latitude")
  input_yaml_multiple(file = config_file, value = lon, key1 = "location", key2 = "longitude")
  input_yaml_multiple(file = config_file, value = surface_elev, key1 = "location", key2 = "elevation")
  input_yaml_multiple(file = config_file, value = Z_max, key1 = "location", key2 = "depth")
  input_yaml_multiple(file = config_file, value = Z_max, key1 = "location", key2 = "init_depth")

  ### Update GLM configuration with lake-specific parameters
  input_yaml_multiple(file = config_file, value = round(sqrt(surface_area), 2), key1 = "model_parameters", key2 = "GLM", key3 = "bsn_len")
  input_yaml_multiple(file = config_file, value = round(sqrt(surface_area), 2), key1 = "model_parameters", key2 = "GLM", key3 = "bsn_wid")
  input_yaml_multiple(file = config_file, value = offset_hours, key1 = "model_parameters", key2 = "GLM", key3 = "timezone")

  ### ------------ Updating configuration with optimal parameters for GLM ------------###
  if (!is.null(best_p)) {
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$light.Kw,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "Kw")
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$mixing.coef_mix_hyp,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "coef_mix_hyp")
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$mixing.coef_wind_stir,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "coef_wind_stir")
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$meteorology.sw_factor,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "sw_factor")
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$meteorology.lw_factor,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "lw_factor")
    input_yaml_multiple(file = config_file,
                        value = best_p$GLM$meteorology.wind_factor,
                        key1 = "model_parameters",
                        key2 = "GLM",
                        key3 = "wind_factor")
  }
}

get_matched_param_suffixes <- function(dir = "cali/") {
  # List GLM files and parameter files (without full paths)
  glm_files <- list.files(path = dir, pattern = "^GLM_LHC_.*\\.csv$", full.names = FALSE)
  param_files <- list.files(path = dir, pattern = "^params_GLM_LHC_.*\\.csv$", full.names = FALSE)

  # Helper function to extract the suffix by removing the prefix and ".csv" extension
  extract_suffix <- function(filename, prefix) {
    suffix <- sub(paste0("^", prefix), "", filename)
    suffix <- sub("\\.csv$", "", suffix)
    return(suffix)
  }

  # Extract suffixes for both file sets
  glm_suffixes <- sapply(glm_files, extract_suffix, prefix = "GLM_LHC_")
  param_suffixes <- sapply(param_files, extract_suffix, prefix = "params_GLM_LHC_")

  # Return the common suffixes between the two sets
  common_suffixes <- intersect(glm_suffixes, param_suffixes)
  return(common_suffixes)
}

# Function to run the GLM simulation for a single lake
run_PB <- function(lake_infor, job, num_cores, config_tmp_root, lswt_dir_root, met_dir_root, tmp_dir_root, output_dir_root) {
  result <- NULL
  lake_id <- lake_infor$LakeID
  output_dir <- file.path(output_dir_root, job, lake_id)

  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  tryCatch({
    # Check if simulation output already exists
    if (!file.exists(file.path(output_dir, 'glm3.json'))) {
      # Create temporary directory for lake-specific run and set working directory
      tmp_dir <- file.path(tmp_dir_root, lake_id)
      dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
      setwd(tmp_dir)

      # Read observed lake surface water temperature data
      observed_lswt <- read_csv(file.path(lswt_dir_root, paste0('lake_id_', lake_id, '.csv')))

      # Determine calibration start and end date based on job type
      if (job == 'exper1') {
        cal_date_end <- as.Date("2014-12-31")
        cal_date_start <- head(observed_lswt$Date, 1) # full year calibration
      } else if (job == 'exper2') {
        valid_data <- observed_lswt[!is.na(observed_lswt$lake_surface_water_temperature), ]
        n <- nrow(valid_data)
        split_index <- round(0.6 * n)
        cal_date_end <- valid_data$Date[split_index]
        cal_date_start <- head(observed_lswt$Date, 1) # full year calibration
      } else {
        cal_date_end <- as.Date("2010-12-31")
        cal_date_start <- cal_date_end %m-% years(3) # 3 year calibration
      }

      meteo_file <- read_csv(file.path(met_dir_root, paste0('met_', lake_id, '.csv')))
      config_tmp <- file.path(config_tmp_root, 'config_tmp.yaml')
      model <- 'GLM'
      config_file <- "config.yaml"
      file.copy(from = config_tmp, to = config_file, overwrite = TRUE)

      # Set simulation start (spin-up for 2 years) and end dates
      # 2 year spin-up
      sim_date_start <- cal_date_start %m-% years(2) # 2 year spin-up
      sim_date_end   <- cal_date_end
      vali_date_end <- tail(observed_lswt$Date, 1)
      # data split
      cal_lswt <- subset(observed_lswt, Date >= cal_date_start & Date <= cal_date_end)
      vali_lswt <- subset(observed_lswt, Date > cal_date_end)
      # Generate standard input files for LakeEnsemblR
      standards <- generate_LER_standard(lake_infor, meteo_file, cal_lswt)
      write_csv(standards[[1]], 'bathymetry_standard.csv')
      write_csv(standards[[2]], 'wtemp_profile_standard.csv')
      write_csv(standards[[3]], 'initial_standard.csv')
      write_csv(standards[[4]], 'meteo_standard.csv')

      # Write and export configuration file
      write_configs(config_file, lake_infor, sim_date_start, sim_date_end)
      export_config(config_file = config_file, model = model)

      cali_res <- cali_ensemble(config_file = config_file, num = 100, cmethod = "LHC",
                                parallel = TRUE, ncores = num_cores, model = model, spin_up = 365*2)
      suffixes <- get_matched_param_suffixes("cali/")
      res_files <- list(
        GLM = list(
          results = paste0("./cali/GLM_LHC_",suffixes,".csv"),
          parameters = paste0("./cali/params_GLM_LHC_",suffixes,".csv")
        )
      )
      res_LHC <- load_LHC_results(config_file = config_file, model = model, res_files = unlist(res_files))
      best_p <- setNames(lapply(model, function(m) res_LHC[[m]][which.min(res_LHC[[m]]$rmse), ]), model)
      best_p_list <- lapply(best_p, function(df) as.list(df))
      best_p_json <- toJSON(best_p_list, pretty = TRUE, auto_unbox = TRUE)
      write(best_p_json, 'cali/best_p.json')
      print("### ------------ cali rmse ------------###")
      write_configs(config_file, lake_infor, sim_date_start, sim_date_end, best_p)
      export_config(config_file = config_file, model = model)
      run_ensemble(config_file = config_file, model = model,
                         parallel = FALSE, ncores = ncores, return_list = FALSE)
      ncdf_dir <- file.path("output/glm_output.nc")
      cali_fit<- calc_fit(ncdf=ncdf_dir, model=model, var = "temp")
      cali_fit_df<- bind_rows(cali_fit, .id = "model")%>%
        mutate(phase = "cali")
      cali_fit_list <- lapply(cali_fit, function(df) as.list(df))
      print("### ------------ vali rmse ------------###")
      standards <- generate_LER_standard(lake_infor, meteo_file, vali_lswt)
      write_csv(standards[[2]], 'wtemp_profile_standard.csv')
      write_configs(config_file, lake_infor, sim_date_start, vali_date_end, best_p)
      export_config(config_file = config_file, model = model)
      run_ensemble(config_file = config_file, model = model,
                         parallel = FALSE, ncores = ncores, return_list = FALSE)
      ncdf_dir <- file.path("output/glm_output.nc")
      vali_fit<- calc_fit(ncdf=ncdf_dir, model=model, var = "temp")
      vali_fit_df<- bind_rows(vali_fit, .id = "model")%>%
        mutate(phase = "vali")
      vali_fit_list <- lapply(vali_fit, function(df) as.list(df))
      message("### ------------ Copying results to output directory ------------###")
      fit_df <- bind_rows(cali_fit_df,vali_fit_df)
      fit_df <- subset(fit_df, model == 'GLM')
      fit_df$LakeID <- lake_id
      fit_df$exper <- job
      write.csv(fit_df, "output/performance_df.csv", row.names = FALSE)

      nml <- read_nml(file.path("GLM/glm3.nml"))
      nml_list <- unclass(nml)


      # file.copy(from = ncdf_dir, to = output_dir, recursive = TRUE)
      file.copy(from = './output', to = output_dir, recursive = TRUE)
      file.copy(from = './cali', to = output_dir, recursive = TRUE)
      file.copy(from = config_file, to = output_dir, recursive = TRUE)
      write_json(nml_list, file.path(output_dir, 'glm3.json'),
                 pretty = TRUE, auto_unbox = TRUE, digits = NA)

      # Clean up temporary directory
      unlink(tmp_dir, recursive = TRUE)
      result <- fit_df
    } else {
      # If output already exists, read and return it
      result <- read_csv(file.path(output_dir, 'output/performance_df.csv'))
      result$LakeID <- lake_id
      result$exper <- job
    }
    return(result)
  }, error = function(e) {
    # Report error and return a default result with NaN values
    message("Error in run_PB: ", conditionMessage(e))
    return(data.frame(
      model = 'GLM',
      rmse = NaN,
      nse = NaN,
      r = NaN,
      bias = NaN,
      mae = NaN,
      nmae = NaN,
      phase = "uncali",
      LakeID = lake_id,
      exper = job
    ))
  })
}

# ------------------------------ Set Root Directories ------------------------------
# root_dir <- 'C:\\Users\\ml101\\MLData\\codes\\PycharmProjects\\trans_PIML'
# job <- 'exper2'

# Retrieve the script directory and move up two levels
script_dir <- getScriptDir()
steps_up <- 2  # number of levels to move up
root_dir <- script_dir
for (i in 1:steps_up) {
  root_dir <- dirname(root_dir)
}

# Set temporary directory based on operating system
if (Sys.info()[["sysname"]] == "Windows") {
  tmp_dir_default <- "D:/tmp/pb"
} else {
  tmp_dir_default <- "/Volumes/S790C/tmp/pb"
}

# Retrieve command-line arguments (job, num_cores, tmp_dir_root)
args <- commandArgs(trailingOnly = TRUE)
cat("Required argument order: job, num_cores, tmp_dir_root\n")
job          <- ifelse(length(args) >= 1, args[1], "exper1") # Options: exper1, exper2, or exper3
num_cores    <- ifelse(length(args) >= 2, as.numeric(args[2]), 10)
tmp_dir_root <- ifelse(length(args) >= 3, args[3], tmp_dir_default)

# Update directories based on the job type
if (job == 'exper1') {
  data_dir_root <- file.path(root_dir, 'data', 'insitulakes')
  lake_list <- read_csv(file.path(data_dir_root, 'lake_info', 'exper1_target.csv'))
} else if (job == 'exper2') {
  data_dir_root <- file.path(root_dir, 'data', 'insitulakes')
  lake_list <- read_csv(file.path(data_dir_root, 'lake_info', 'exper2_target.csv'))
} else {
  data_dir_root <- file.path(root_dir, 'data', 'ccilakes')
  lake_list <- read_csv(file.path(data_dir_root, 'lake_info', 'ESA_CCI.csv'))
}
config_tmp_root <- file.path(root_dir, 'model', 'Benchmarks', 'PB')
lswt_dir_root   <- file.path(data_dir_root, 'lswts')
met_dir_root    <- file.path(data_dir_root, 'mets')
output_dir_root <- file.path(root_dir, 'result', 'PB')
if (!dir.exists(output_dir_root)) {
  dir.create(output_dir_root, recursive = TRUE)
}

# ------------------------------ Parallel Execution ------------------------------
# Pre-split lake_list into a list of lake information objects by LakeID
lake_info_list <- split(lake_list, lake_list$LakeID)
results<-{}
i<-1
# lake_infor <- lake_info_list[[41]]
for (lake_infor in lake_info_list){
  result <-
    run_PB(
      lake_infor = lake_infor,
      job = job,
      num_cores = num_cores,
      config_tmp_root = config_tmp_root,
      lswt_dir_root = lswt_dir_root,
      met_dir_root = met_dir_root,
      tmp_dir_root = tmp_dir_root,
      output_dir_root = output_dir_root
  )
  results[[i]]<-result
  i<-i+1
}

# Combine all results into a single data frame and write to CSV
all_results <- do.call(rbind, results)
write.csv(all_results, file.path(output_dir_root, job, 'GLM.csv'), row.names = FALSE)