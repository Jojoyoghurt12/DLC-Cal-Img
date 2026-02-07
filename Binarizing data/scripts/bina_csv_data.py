import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os

def extract_binary(calcium_trace, sampling_frequency, z_threshold):
    """
    Converts raw calcium traces into binary traces
    
    Parameters:
    calcium_trace: array-like, activity of a single neuron
    sampling_frequency: float, frequency at which calcium imaging was done (Hz or fps)
    z_threshold: float, standard deviation threshold above which activity is considered signal
    
    Returns:
    binarized_trace: binary array (0s and 1s)
    filtered_trace: low-pass filtered trace
    norm_trace: normalized trace (z-score)
    d1_trace: first derivative of filtered trace
    """
    
    # Design butterworth filter (2nd order, 2 Hz cutoff)
    nyquist = sampling_frequency / 2
    cutoff = 2 / nyquist
    b_filt, a_filt = butter(2, cutoff, btype='low')
    
    # Apply filter
    filtered_trace = filtfilt(b_filt, a_filt, calcium_trace)
    
    # Normalize by standard deviation (z-score)
    norm_trace = filtered_trace / np.std(filtered_trace)
    
    # Calculate first derivative
    d1_trace = np.diff(filtered_trace)
    d1_trace = np.append(d1_trace, 0)  # Add zero at end to match length
    
    # Create binary trace: active when above threshold AND rising
    binarized_trace = np.zeros_like(calcium_trace)
    binarized_trace[(norm_trace > z_threshold) & (d1_trace > 0)] = 1
    
    return binarized_trace, filtered_trace, norm_trace, d1_trace

def process_calcium_data(csv_path, sampling_frequency=30, z_threshold=2.0):
    """
    Process all cells in the calcium data CSV file

    Input:
    - .csv file made by convert_to_csv

    Output:
    - binary df
    - filtered df
    - normalized df
    - derivative df
    
    Parameters:
    csv_path: path to C_wide.csv file
    sampling_frequency: imaging frequency in Hz/fps
    z_threshold: threshold for binarization (typically 2-3)
    """
    
    # Load the wide format data
    df = pd.read_csv(csv_path)
    
    # Get cell columns (all except 'frame')
    cell_columns = [col for col in df.columns if col.startswith('cell_')]
    
    # Create output dataframes
    df_binary = df[['frame']].copy()
    df_filtered = df[['frame']].copy()
    df_normalized = df[['frame']].copy()
    df_derivative = df[['frame']].copy()

    # Process each cell
    for cell_col in cell_columns:
        print(f"Processing {cell_col}...")
        
        calcium_trace = df[cell_col].values
        
        # Apply binarization
        binary, filtered, normalized, derivative = extract_binary(
            calcium_trace, sampling_frequency, z_threshold
        )
        
        # Store results
        df_binary[f'{cell_col}_binary'] = binary
        df_filtered[f'{cell_col}_filtered'] = filtered
        df_normalized[f'{cell_col}_normalized'] = normalized
        df_derivative[f'{cell_col}_derivative'] = derivative
        
        # Print some stats
        active_frames = np.sum(binary)
        total_frames = len(binary)
        activity_rate = (active_frames / total_frames) * 100
        print(f"  {cell_col}: {active_frames}/{total_frames} active frames ({activity_rate:.1f}%)")
    
    return df_binary, df_filtered, df_normalized, df_derivative

if __name__ == "__main__":
    # Parameters - adjust these as needed
    SAMPLING_FREQUENCY = 30  # Hz or fps - adjust based on your imaging rate
    Z_THRESHOLD = 2.0        # Standard deviations above noise (2-3 is typical)
    
    # File paths
    input_file = "../test videos/minian/C.csv"
    output_dir = "../test videos/minian/"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please run the convert_to_csv.py script first to create C.csv")
    
    # Process the data
    try:
        df_binary, df_filtered, df_normalized, df_derivative = process_calcium_data(
            input_file, SAMPLING_FREQUENCY, Z_THRESHOLD
        )
        
        # Save results
        df_binary.to_csv(os.path.join(output_dir, "C_binary.csv"), index=False)
        df_filtered.to_csv(os.path.join(output_dir, "C_filtered.csv"), index=False)
        df_normalized.to_csv(os.path.join(output_dir, "C_normalized.csv"), index=False)
        df_derivative.to_csv(os.path.join(output_dir, "C_derivative.csv"), index=False)
        
        print("\nSaved files:")

        
    except Exception as e:
        print(f"Error processing data: {e}")