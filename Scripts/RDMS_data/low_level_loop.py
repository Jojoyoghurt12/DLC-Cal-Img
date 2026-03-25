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
import importlib.util
from pathlib import Path, PurePosixPath

# Add the Minian_data folder to sys.path so pipeline.py can be imported
# regardless of the current working directory when this script is run.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINIAN_SCRIPTS_DIR = os.path.join(_THIS_DIR, "..", "Minian_data")
if _MINIAN_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MINIAN_SCRIPTS_DIR))

_PIPELINE_FILE = os.path.join(_MINIAN_SCRIPTS_DIR, "pipeline.py")
_pipeline_spec = importlib.util.spec_from_file_location("pipeline", _PIPELINE_FILE)
if _pipeline_spec is None or _pipeline_spec.loader is None:
    raise ImportError(f"Could not load pipeline module from {_PIPELINE_FILE}")
pipeline = importlib.util.module_from_spec(_pipeline_spec)
_pipeline_spec.loader.exec_module(pipeline)

# local destination using ils command and cd to your folder
IRODS_BASE   = "/rug/home/ATLAS_lab/Jesse_ten_Broeke" 
SCRATCH_BASE = Path("/scratch/s4750098")   # local scratch destination
OUTPUT_BASE  = Path("/scratch/s4750098/minian_outputs")

# List of folder names that live inside IRODS_BASE and you want to look through for data
# Use path if folders are nested
# These are the superceding folders, containing the individual nested test folders


FOLDER_LIST = ["Batch 1/24"]

# Here you can filter through the nested folders for specifically only the ones you want to analyze.
REQUIRED = ("_T", "_PostEx")

# Optional narrower subset within the required filter.
# Example: ("_T1", "_T2") to process only those sessions.
T_SUBSET = ()

# Function for looking through the total on iRODS_BASE before downloading. 
def discover_folders_under(root_rel: str) -> list[PurePosixPath]:
    root_abs = f"{IRODS_BASE}/{root_rel}"
    # Run the ils command with flag -r to get a recursive list of the root folder.
    result = subprocess.run(["ils", "-r", root_abs], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    # Parse the output to extract folder paths relative to IRODS_BASE. The output lines that end with ":" indicate folders.
    folders = []
    for line in result.stdout.splitlines():
        s = line.strip()
        if not s.endswith(":"):
            continue
        s = s[:-1]
        if s.startswith("C- "):
            s = s[3:].strip()
        p = PurePosixPath(s)
        try:
            rel = p.relative_to(PurePosixPath(IRODS_BASE))
        except ValueError:
            continue
        folders.append(rel)

    return folders

def select_folders_for_download(folders: list[PurePosixPath]) -> list[PurePosixPath]:
    selected: list[PurePosixPath] = []
    for folder in folders:
        name = folder.name
        if not all(token in name for token in REQUIRED):
            continue
        if T_SUBSET and not any(tag in name for tag in T_SUBSET):
            continue
        selected.append(folder)

    # Keep a stable, duplicate-free order across runs.
    return sorted(set(selected), key=str)


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

def run_computation(local_folder: Path, output_folder: Path) -> None:
    """
    Run the full Minian CNMF pipeline on the downloaded session folder,
    then run the post-processing scripts (convert_to_csv, plotting, map).
    local_folder is the Path to the downloaded data on /scratch.
    """
    print(f"[compute] Starting Minian pipeline on {local_folder} ...")
    pipeline.run_pipeline(str(local_folder), output_dir=str(output_folder))
    print(f"[compute] Pipeline finished for {local_folder}")



# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    all_folders: list[PurePosixPath] = []
    for root in FOLDER_LIST:
        all_folders.extend(discover_folders_under(root))

    dwnld_folders = select_folders_for_download(all_folders)
    if not dwnld_folders:
        print(f"[info] No matching folders found under {FOLDER_LIST}")
        return

    print(f"[info] Found {len(dwnld_folders)} folder(s) to process")

    for folder_name in dwnld_folders:
        irods_path = f"{IRODS_BASE}/{folder_name.as_posix()}"
        local_folder = SCRATCH_BASE / Path(*folder_name.parts)
        output_folder = OUTPUT_BASE / folder_name.name
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*60}")

        try:
            iget(irods_path, local_folder)
            run_computation(local_folder, output_folder)
        except Exception as e:
            print(f"[ERROR] {folder_name}: {e}")
        finally:
            # Always clean up local scratch, even if computation failed
            cleanup_local(local_folder)

    print("\n[done] All folders processed.")


if __name__ == "__main__":
    main()
