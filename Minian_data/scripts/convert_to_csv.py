import os
import xarray as xr

def convert_to_csv(param_save_minian, file_name):
    """
    Goal : Convert .zarr files to a more easily readable csv per cell per frame

    Input:
    - param_save_minian: dict with key "dpath" for directory path

    Output: 
    - C.csv: csv file with columns frame, activity, per cell

    Parameters:
    - out_dir: directory where the .csv file will be saved. 
    - c_path: path to C.zarr file
    - df_wide.columns: renaming columns, can be adjusted. 
        Otherwise uses unit_id
    """
    # Use the same output folder as the pipeline
    out_dir = param_save_minian["dpath"]
    
    # Check for directory
    if not os.path.exists(out_dir):
        print(f"Output directory does not exist: {out_dir}")
        return
    
    # List available zarr files
    zarr_files = [f for f in os.listdir(out_dir) if f.endswith('.zarr')]
    print(f"Available zarr files in {out_dir}: {zarr_files}")
    
    # Temporal activity C
    c_path = os.path.join(out_dir, "C.zarr")
    
    if not os.path.exists(c_path):
        print(f"C.zarr not found at {c_path}")
        return
    
    try:
        C = xr.open_zarr(c_path, consolidated=False)
        print(f"Successfully opened C.zarr with shape: {C.dims}")
        
        # Convert to a tidy table and save
        if hasattr(C, 'data_vars') and len(C.data_vars) > 0:
            # Get the data variable
            var_name = list(C.data_vars)[0]
            data_array = C[var_name]
            print(f"Using data variable: {var_name}")
        else:
            # If it's already a DataArray
            data_array = C
        
        if file_name == None:
                file_name = "C.csv"
        # Convert to DataFrame (long format) - necessary for proper pivot
        df_long = data_array.to_dataframe().reset_index()
        # Each cell becomes a column
        df_wide = df_long.pivot(index='frame', columns='unit_id', values=df_long.columns[-1])
        # Rename columns to cell names - can be adjusted
        df_wide.columns = [f'cell_{col}' for col in df_wide.columns]
        # Reset index so frame becomes a regular column for later use
        df_wide = df_wide.reset_index()
        # Save
        csv_path = os.path.join(out_dir, file_name)
        df_wide.to_csv(csv_path, index=False)

    # Error check    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # adjust path to your .zarr file
    param_save_minian = {"dpath": "./test videos/minian"}
    file_name = None # or specify a file name, else "C.csv"
    convert_to_csv(param_save_minian, file_name)