"""
iRODS data iteration
-------------------------------
For each folder in FOLDER_LIST:
  1. iget  : download from iRODS -> /scratch
  2. compute: run your analysis on the local copy
  3. cleanup: delete the local /scratch copy (iRODS copy is untouched)

Requirements
  - icommands installed and iinit authenticated before running (Hábrok has it preinstalled)
  - /scratch writable (depends on your own settings)

----------------------------------------------------------------------------------------------------
  Make sure to be connected to the RDMS server;
  login using :     ssh s123456@interactive1.hb.hpc.rug.nl (if using VScode interactive server)
                    ssh s123456@login1.hb.hpc.rug.nl (if using cluster)
  Enter your password + Authenticator code
  See eagle 
  (for first time use)
  check for irods_environment.json file existence - ls~/.irods/
  make sure your name is also the user email - nano ~/.irods/irods_environment.json
    save using Ctrl-o and exit using Ctrl-x or Ctrl-c
  Initiate "mount" on RDMS - iinit
  you can use ihelp for functions if you want. 
  use ils to look at your root folder base, this is necessary in the IRODS_BASE parameter
  also make sure SCRATCH_BASE = Path("/scratch/s123456") to make sure you write to your own scratch folder

  Then edit FOLDER_LIST to your folder names and directories for download. 
----------------------------------------------------------------------------------------------------
"""

import subprocess
import shutil
import sys
import os
from pathlib import Path

# Add the Minian_data folder to sys.path so pipeline.py can be imported
# regardless of the current working directory when this script is run.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINIAN_SCRIPTS_DIR = os.path.join(_THIS_DIR, "..", "Minian_data")
if _MINIAN_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MINIAN_SCRIPTS_DIR))

import pipeline

# local destination using ils command and cd to your folder
IRODS_BASE   = "/rug/home/j.ten.broeke.1@student.rug.nl/MiResearch Project S.C.W. van der Veldt" 
SCRATCH_BASE = Path("/scratch/s4750098")   # local scratch destination

# List of folder names that live inside IRODS_BASE and you want to download
# Use path if folders are nested
FOLDER_LIST = [
    "session_001",
    "session_002",
    "session_003",
]



def iget(irods_path: str, local_path: Path) -> None:
    """Download a collection from iRODS to local_path using iget."""
    local_path.mkdir(parents=True, exist_ok=True)
    # run the iget function in terminal using subprocess, together with the -r (progress) and -f (force) flags
    result = subprocess.run(
        ["iget", "-r", "-f", irods_path, str(local_path)],
        capture_output=True,
        text=True,
    )
    # Check if the command was successful, if not raise an error with the stderr output
    if result.returncode != 0:
        raise RuntimeError(
            f"iget failed for {irods_path}\n"
            f"stderr: {result.stderr.strip()}"
        )
    print(f"[iget] Downloaded {irods_path} -> {local_path}")


def cleanup_local(local_path: Path) -> None:
    """Remove the local /scratch copy. iRODS copy is NOT touched."""
    # Only run after you finished processes are done, data should be kept. 
    # Make sure to write output data to a seperate folder if you want to keep it, otherwise it will be deleted as well.
    if local_path.exists():
        shutil.rmtree(local_path)
        print(f"[cleanup] Removed local copy: {local_path}")


# ---------------------------------------------------------------------------
# COMPUTATION  – replace / extend this with your actual analysis
# ---------------------------------------------------------------------------

def run_computation(local_folder: Path) -> None:
    """
    Run the full Minian CNMF pipeline on the downloaded session folder,
    then run the post-processing scripts (convert_to_csv, plotting, map).
    local_folder is the Path to the downloaded data on /scratch.
    """
    print(f"[compute] Starting Minian pipeline on {local_folder} ...")
    pipeline.run_pipeline(str(local_folder))
    print(f"[compute] Pipeline finished for {local_folder}")



# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    for folder_name in FOLDER_LIST:
        irods_path  = f"{IRODS_BASE}/{folder_name}"
        local_folder = SCRATCH_BASE / folder_name
        pipeline.MINIAN_PATH = {folder_name: str(local_folder)}  # Set the minian path for this folder"

        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*60}")

        try:
            iget(irods_path, local_folder)
            run_computation(local_folder)
        except Exception as e:
            print(f"[ERROR] {folder_name}: {e}")
        finally:
            # Always clean up local scratch, even if computation failed
            cleanup_local(local_folder)

    print("\n[done] All folders processed.")


if __name__ == "__main__":
    main()
