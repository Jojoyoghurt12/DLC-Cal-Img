'''
Check for all frames in the DLC output data whether the mouse is in the ROI or not. 
if so, save the frame number in a list.
export the list of frames to check for sysclock later,
'''

import argparse
import pandas as pd
from pathlib import Path

def roi_check(dlc_csv_path: Path, roi_coords: tuple([int, int, int, int])) -> list[int]: # output list is integer matching framenumbers
    """
    Goal:
    Check for all frames in the DLC output data whether the mouse is in the ROI or not.

    Parameters:
    - dlc_csv_path: Path to the DLC output CSV file containing frame numbers and coordinates.
    - roi_coords: Tuple of (x, y, w, h) defining the rectangular ROI.
        x = top-left x-coordinate of the ROI
        y = top-left y-coordinate of the ROI
        w = width of the ROI
        h = height of the ROI
    
    Output:
    - List of frame numbers where the mouse is detected within the ROI.
    """

    # load the DLC output with multi-level headers
    dlc_data = pd.read_csv(dlc_csv_path, header=[0, 1, 2])
    
    # Find columns that have "red_light" in bodyparts (level 1) and "x" or "y" in coords (level 2)
    red_light_xy_cols = []
    for col in dlc_data.columns:
        bodypart = col[1]  # Level 1 - bodyparts
        coord = col[2]     # Level 2 - coordinates
        if "red_light" in bodypart and coord in ["x", "y"]:
            red_light_xy_cols.append(col)
    
    # Extract the specific red_light x and y columns
    red_light_x_col = None
    red_light_y_col = None
    
    for col in red_light_xy_cols:
        if col[2] == "x":
            red_light_x_col = col
        elif col[2] == "y":
            red_light_y_col = col
    
    if red_light_x_col is None or red_light_y_col is None:
        raise ValueError("Could not find red_light x and y coordinates in the data")

    # ROI coordinates
    roi_x, roi_y, w, h = roi_coords

    # calculate the bottom-right corner of the ROI
    roi_x2 = roi_x + w
    roi_y2 = roi_y + h

    # list to store frame numbers where the mouse is in the ROI
    frames_in_roi = []

    # iterate through each row in the DLC data
    for index, row in dlc_data.iterrows():
        # Get the red_light coordinates from this frame
        mouse_x = row[red_light_x_col]
        mouse_y = row[red_light_y_col]
        
        # Check if the coordinates are within the ROI bounds
        if (roi_x <= mouse_x <= roi_x2) and (roi_y <= mouse_y <= roi_y2):
            frames_in_roi.append(index)
    
    return frames_in_roi



if __name__ == "__main__":
    dlc_csv_path = Path(r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\DLC_data\to csv\combined_dlc_data.csv")  # replace with actual path to DLC output CSV file
    roi_coords = (167, 145, 133, 176)  # paste tuple that was the output of draw_ROI.py
    
    frames_in_roi = roi_check(dlc_csv_path, roi_coords)
    print(f"Found {len(frames_in_roi)} frames with red_light coordinates in ROI")
    print(f"Frame numbers in ROI: {frames_in_roi[:10]}{'...' if len(frames_in_roi) > 10 else ''}")
