import os
import subprocess

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
# Path to the Python script you want to run (update filename if different)
R_script = os.path.join(root_dir, 'model', 'run_scr', "run_glm_cali_valid.R")
task_ids = ['WindermereSouthBasin', 'Feeagh', 'Kasumigaura']
# task_ids = ['WindermereSouthBasin']
for task_id in task_ids:
    # Build the command line call
    cmd = [
        "Rscript",
        R_script,
        task_id,
        str(8)
    ]
    # Print the command for debugging purposes and run it
    # print("Running command:", " ".join(cmd))
    print("Running command for:", task_id)
    subprocess.run(cmd, env=env, cwd=root_dir)