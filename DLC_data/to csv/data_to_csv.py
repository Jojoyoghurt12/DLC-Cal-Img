# Structure of data from DLC is as follows:
"""
Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 | ....

scorer | name of model used 
bodyparts   | body part 1   | body part 1   | likelihood of body part 1     | body part 2   | body part 2 | ... 
Frame 1     | x bp 1        | y bp 1        | likelihood bp 1               | x bp 2        | y bp 2      | ...
Frame 2     | x bp 1        | y bp 1        | likelihood bp 1               | x bp 2        | y bp 2      | ...
...
"""

import pandas as pd
import numpy as np 
import os

def combine_all_csv(folder_path, output_file):
    """
    Combines all .h5 files into a single csv file.

    inputs:
    -folder path: str, path to the folder containing the .h5 files.
        ( can be modified to dynamically look for .h5 files in multiple folders if needed)
    -output_file: str, name of the output csv file.

    outputs: 
    Creates a new csv file with all the data combined, named after output_file parameter.

    """

    # find all .h5 files in the folder, and organize them chronologically in a list
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    # sort them alphabetically (video names should be made alphabetical in .\names\dlc_data_name)
    h5_files.sort()

    # to detect directory issues
    if not h5_files:
        print("No .h5 files found in the specified folder.")
        return
    
    # set empty list to hold dataframes
    dfs = []

    # load all the .h5 files 
    for file in h5_files:
        file_path = os.path.join(folder_path, file)
        # creates a dataframe for each individual file
        df = pd.read_hdf(file_path)
        dfs.append(df)
        # maybe add an exception handler in case of read errors

    # combine all dfs into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # csv conversion
    if output_file is None:
        output_file = "combined_dlc_data.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as {output_file}")

if __name__ == "__main__":
    folder_path = "C:\\Users\\Gebruiker\\Desktop\\MiPro_2026\\cleaning data\\dlc data expl"
    output_file = None  # or specify a name like "combined_dlc_data.csv"
    combine_all_csv(folder_path, output_file)

