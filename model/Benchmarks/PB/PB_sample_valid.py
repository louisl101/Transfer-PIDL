import os
import subprocess
import pandas as pd
# Copy the current environment variables and set R-related paths
env = os.environ.copy()

# Set R environment variables to match RStudio
env["R_LIBS_USER"] = "C:\\Users\\ml101\\AppData\\Local\\R\\win-library\\4.4"
env["R_LIBS_SITE"] = "C:/PROGRA~1/R/R-44~1.2/site-library"
env["R_HOME"] = "C:/PROGRA~1/R/R-44~1.2"
env["R_ARCH"] = "/x64"
env["PATH"] = "C:\\PROGRA~1\\R\\R-4.4.2\\bin\\x64;" + env["PATH"]
env["R_RTOOLS44_PATH"] = "c:/rtools44/x86_64-w64-mingw32.static.posix/bin;c:/rtools44/usr/bin"
env["RSTUDIO_PANDOC"] = "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools"
env["RSTUDIO"] = "1"

# Set the project root directory (adjust as needed)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(root_dir, 'result', 'PB', 'exper1', 'sample_valid')
# Path to the Python script you want to run (update filename if different)
R_script = os.path.join(root_dir, 'model', 'run_scr', "run_glm_sample_valid.R")
task_ids = ['WindermereSouthBasin', 'Feeagh', 'Kasumigaura']
# fine_tune_sizes = [i*5 for i in [1, 2, 4, 8, 16, 32, 64, 128]]  # number or fraction of points to retain for fine-tuning (< 0 for all, 0-1 for percentage, >= 1 absolute number)
fine_tune_sizes = [i * 30 for i in [1, 3, 6, 12, 18, 24, 30, 36, 42, 48]]  # number or fraction of points to retain for fine-tuning (< 0 for all, 0-1 for percentage, >= 1 absolute number)
iter_seeds = [10, 20, 30, 40, 50, 60]  # seed to fix the random sampling of pretrain_scale and fine_tune_size, useless during fine-tuning
aggregated_results = []
for task_id in task_ids:
    for fs in fine_tune_sizes:
        for iter_seed in iter_seeds:
            # Build the command line call
            cmd = [
                "Rscript",
                R_script,
                task_id,
                str(10),
                str(fs),
                str(iter_seed)
            ]
            # Print the command for debugging purposes and run it
            result_dir = os.path.join(output_dir, task_id, f'fs_{fs}', f'iter_seed_{iter_seed}')
            result_file = os.path.join(result_dir, 'output','performance_df.csv')
            if not os.path.isfile(result_file):
                print("Running command for:", task_id)
                subprocess.run(cmd, env=env, cwd=root_dir)

            # Read result and add extra columns
            result = pd.read_csv(result_file)
            result['method'] = 'PB'
            aggregated_results.append(result)
            # Concatenate all aggregated results and save to a CSV file
            aggregated_df = pd.concat(aggregated_results, ignore_index=True)
            aggregated_csv_path = os.path.join(output_dir, "PB_sample_valid.csv")
            aggregated_df.to_csv(aggregated_csv_path, index=False)