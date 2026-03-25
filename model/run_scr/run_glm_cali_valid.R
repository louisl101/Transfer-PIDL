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
# Function to get the current script's directory when using Rscript non-interactively
getScriptDir <- function() {
  # Get all command-line arguments
  args <- commandArgs(trailingOnly = FALSE)
  # Find the argument that starts with '--file='
  fileArg <- grep("^--file=", args, value = TRUE)

  if (length(fileArg) > 0) {
    # Remove the '--file=' prefix and normalize the path
    scriptPath <- normalizePath(sub("^--file=", "", fileArg))
    return(dirname(scriptPath))
  } else {
    stop("Unable to determine script directory. Are you running this using Rscript?")
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
## --------------------------reproduce----------------------------##
model <- "GLM"
if (Sys.info()[["sysname"]] == "Windows") {
    # Use forward slashes in R, even on Windows.
    tmp_dir_default <- "D:/tmp/pb"
  } else {
    # Example path on macOS (adjust to your actual mount point).
    tmp_dir_default <- "/Volumes/S790C/tmp/pb"
  }
## --------------------------Retrieve command-line arguments (trailingOnly = TRUE returns only user-supplied arguments)----------------------------##
args <- commandArgs(trailingOnly = TRUE)
cat("Required argument order: lake_name, ncores, tmp_dir_root\n")
lake_name    <- ifelse(length(args) >= 1, args[1], "WindermereSouthBasin")
ncores       <- ifelse(length(args) >= 2, as.numeric(args[2]), 10)
tmp_dir <- ifelse(length(args) >= 3, args[3], tmp_dir_default)
# Print the parameters for confirmation
cat("tmp_dir:", tmp_dir, "\n")
cat("ncores:", ncores, "\n")
cat("lake_name:", lake_name, "\n")
cat("model:", model, "\n")
# lake_name <- "Feeagh"
# script_dir <-"/Users/ml/Documents/ML-data/9-CODES/PycharmProjects/trans_PIML/model/run_src"
## --------------------------base file path ----------------------------##
# the current script's directory
script_dir <- getScriptDir()
steps_up <- 2  # number of levels to move up
root_dir <- script_dir
for (i in 1:steps_up) {
  root_dir <- dirname(root_dir)
}
print(root_dir)
file_folder <- file.path(root_dir,'data', 'insitu_valid', lake_name)
output_dir <- file.path(root_dir, 'result', 'PB', 'exper1' ,lake_name)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
## --------------------------starts----------------------------##
print("### ------------ Workflow starts! ------------###")
print("### ------------ Copying data folder to temporary directory ------------###")
if (!dir.exists(tmp_dir)) {
  dir.create(tmp_dir, recursive = TRUE)
}
file.copy(from = file_folder, to = tmp_dir, recursive = TRUE)
setwd(file.path(tmp_dir, lake_name))  # Change working directory to the lake folder
print("### ------------ pre-processing data ------------###")
observed_lswt <- read.csv("wtemp_profile_standard.csv", header = TRUE, stringsAsFactors = FALSE)
# observed_lswt$datetime <- as.POSIXct(observed_lswt$datetime)
cal_date_end <- as.Date("2014-12-31")
cal_date_start <- head(observed_lswt$datetime, 1) # full year calibration
# Set simulation start (spin-up for 2 years) and end dates
# 2 year spin-up
sim_date_start <- as.Date(cal_date_start) %m-% years(2) # 2 year spin-up
sim_date_end   <- cal_date_end
vali_date_end <- tail(observed_lswt$datetime, 1)
# data split
cal_lswt <- subset(observed_lswt, datetime >= cal_date_start & datetime <= cal_date_end)
vali_lswt <- subset(observed_lswt, datetime > cal_date_end)
print("### ------------ calibrating ------------###")
write_csv(cal_lswt, file = "wtemp_profile_standard.csv")
config_file <- "conifg.yaml"
input_yaml_multiple(file = config_file, value = paste0(sim_date_start, " 00:00:00"), key1 = "time", key2 = "start")
input_yaml_multiple(file = config_file, value = paste0(cal_date_end, " 00:00:00"), key1 = "time", key2 = "stop")
export_config(config_file = config_file, model = model)
#
cali_res <- cali_ensemble(config_file = config_file, num = 100, cmethod = "LHC",
                          parallel = TRUE, ncores = ncores, model = model, spin_up = 365*2)
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
input_yaml_multiple(file = config_file, value = paste0(sim_date_start, " 00:00:00"), key1 = "time", key2 = "start")
input_yaml_multiple(file = config_file, value = paste0(cal_date_end, " 00:00:00"), key1 = "time", key2 = "stop")
#
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
export_config(config_file = config_file, model = model)
run_ensemble(config_file = config_file, model = model,
                         parallel = FALSE, ncores = ncores, return_list = FALSE)
ncdf_dir <- file.path("output/glm_output.nc")
cali_fit<- calc_fit(ncdf=ncdf_dir, model=model, var = "temp")
cali_fit_df<- bind_rows(cali_fit, .id = "model")%>%
  mutate(phase = "cali")
cali_fit_list <- lapply(cali_fit, function(df) as.list(df))
print("### ------------ vali rmse ------------###")
write_csv(vali_lswt, file = "wtemp_profile_standard.csv")
input_yaml_multiple(file = config_file, value = paste0(sim_date_start, " 00:00:00"), key1 = "time", key2 = "start")
input_yaml_multiple(file = config_file, value = paste0(vali_date_end, " 00:00:00"), key1 = "time", key2 = "stop")
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
fit_df$LakeID <- lake_name
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
