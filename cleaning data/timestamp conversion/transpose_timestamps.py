#!/usr/bin/env python3
"""
Transpose timestamp.dat file into a more intuitive paired format.
Converts from interleaved camera records to paired frame records.
"""

import pandas as pd
from pathlib import Path

def transpose_timestamps(input_file, output_file=None):
    """
    Goal: Read interleaved camera timestamps and create paired frame rows.
    Detects dropped frames by showing NaN when frame numbers don't align.
    
    Input format (interleaved):
        camNum | frameNum | sysClock | buffer

    
    Output format (paired):
        framenumber | cam0_frame | cam0_sysclock | cam2_frame | cam2_sysclock

    """
    
    # Input reading
    df = pd.read_csv(input_file, sep='\t')
    
    # Separate data by camera
    cam0_data = df[df['camNum'] == 0].reset_index(drop=True)
    cam2_data = df[df['camNum'] == 2].reset_index(drop=True)
    
    # Create dictionaries for by framenumber lookup
    cam0_dict = dict(zip(cam0_data['frameNum'], cam0_data['sysClock']))
    cam2_dict = dict(zip(cam2_data['frameNum'], cam2_data['sysClock']))
    
    # Get all unique frame numbers across both cameras
    all_frames = sorted(set(cam0_data['frameNum'].tolist() + cam2_data['frameNum'].tolist()))
    
    # Build paired data with missing frames shown as NaN
    paired_data = {
        'framenumber': all_frames,
        'cam0_frame': [f if f in cam0_dict else None for f in all_frames],
        'cam0_sysclock': [cam0_dict.get(f, None) for f in all_frames],
        'cam2_frame': [f if f in cam2_dict else None for f in all_frames],
        'cam2_sysclock': [cam2_dict.get(f, None) for f in all_frames],
    }
    
    result_df = pd.DataFrame(paired_data)
    
    # Replace None with 'NaN' for display in CSV
    result_df = result_df.fillna('NaN')
    
    # Write output as stem+_paired.csv if no output file provided
    if output_file is None:
        output_file = Path(input_file).stem + '_paired.csv'
    
    # print total frames and header
    result_df.to_csv(output_file, index=False)
    print(f"✓ Transposed data written to: {output_file}")
    print(f"  Total unique frames: {len(result_df)}")
    print(result_df.head(15).to_string(index=False))
    
    return result_df

if __name__ == '__main__':
    input_file = r'.\timestamp.dat'
    output_file = r'.\timestamp_paired.csv'
    
    transpose_timestamps(input_file, output_file)
