import pandas as pd
import numpy as np


def load_dlc_data(file_path):
    """"
    Parameters:
    - file_path: str, path to the .h5 file containing DLC data.
    - scorer: str, the scorer name used in the DLC project.

    Returns:
    - df: pandas DataFrame containing the DLC data.
    """
    # Loading DLC data to analyse
    df = pd.read_hdf(file_path)
    if df.empty:
        raise ValueError("The loaded DataFrame is empty. Please check the file path and scorer name.")
    return df
# Can add variable for the folder location in case of'

# convert first 10 rows to csv for quick inspection of structure
def convert_to_csv(df, output_file=None):
    if output_file is None:
        output_file = "dlc_data_sample.csv"
    df.head(20).to_csv(output_file, index=False)

if __name__ == "__main__":
    df = load_dlc_data('Nol A1rl Test 070324DLC_resnet50_BN research projectFeb12shuffle1_205000 (1).h5')
    convert_to_csv(df)
    print(df.head())