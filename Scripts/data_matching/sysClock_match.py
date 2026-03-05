import pandas as pd
import numpy as np
import os


def find_closest_cam2_frame(cam0_sysclock_val: float, cam2_clocks, cam2_frames) -> int:
    """
    Return the cam2_frame whose sysclock is closest to cam0_sysclock_val.
    
    Parameters:
    cam0_sysclock_val: float, system clock value from cam0
    cam2_clocks: array, sorted cam2 sysclock values
    cam2_frames: array, corresponding cam2 frame numbers
    
    Returns:
    int, the cam2_frame with closest sysclock value
    """
    idx = np.searchsorted(cam2_clocks, cam0_sysclock_val)
    # Check neighbours
    candidates = []
    for i in [idx - 1, idx]:
        if 0 <= i < len(cam2_clocks):
            candidates.append((abs(cam2_clocks[i] - cam0_sysclock_val), cam2_frames[i]))
    return min(candidates, key=lambda x: x[0])[1]


def match_cam0_to_cam2_frames(timestamp_path, c_binary_path):
    """
    Match C_binary calcium imaging frames to DLC camera frames using system clock synchronization.
    
    Maps each calcium imaging frame (cam0) to the nearest DLC frame (cam2) based on system clock 
    timestamps. For each cell, collects which cam2 frames were active (binary == 1).
    
    Parameters:
    timestamp_path: str, path to timestamp_paired.csv file
    c_binary_path: str, path to C_binary.csv file
    
    Returns:
    DataFrame with columns: cell, active_cam2_frames, n_active_frames
    int, number of frames skipped due to missing timestamps
    """
    
    # Load data
    timestamps = pd.read_csv(timestamp_path)
    c_binary = pd.read_csv(c_binary_path)
    
    # Clean up timestamp table - drop rows where either sysclock column is NaN
    timestamps = timestamps.dropna(subset=["cam0_frame", "cam0_sysclock", "cam2_frame", "cam2_sysclock"])
    timestamps["cam0_frame"] = timestamps["cam0_frame"].astype(int)
    timestamps["cam0_sysclock"] = timestamps["cam0_sysclock"].astype(float)
    timestamps["cam2_frame"] = timestamps["cam2_frame"].astype(int)
    timestamps["cam2_sysclock"] = timestamps["cam2_sysclock"].astype(float)
    
    # Build fast lookup: cam0_frame → cam0_sysclock
    # C_binary uses 0-based frame index; cam0_frame in timestamps is 1-based.
    cam0_lookup = timestamps.set_index("cam0_frame")["cam0_sysclock"].to_dict()
    
    # Pre-sort cam2 sysclock values for searchsorted
    cam2_sorted = timestamps[["cam2_frame", "cam2_sysclock"]].sort_values("cam2_sysclock").reset_index(drop=True)
    cam2_clocks = cam2_sorted["cam2_sysclock"].values
    cam2_frames = cam2_sorted["cam2_frame"].values
    
    # Build frame → cam2_frame lookup for all C_binary frames
    cell_cols = [c for c in c_binary.columns if c != "frame"]
    
    skipped = 0
    frame_to_cam2 = {}
    
    for _, row in c_binary.iterrows():
        cam0_frame_id = int(row["frame"]) + 1  # C_binary is 0-based; timestamps are 1-based
        cam0_sysclock = cam0_lookup.get(cam0_frame_id)
        if cam0_sysclock is None:
            skipped += 1
            continue
        frame_to_cam2[int(row["frame"])] = find_closest_cam2_frame(
            cam0_sysclock, cam2_clocks, cam2_frames
        )
    
    # For each cell, collect cam2_frames where it was active (binary == 1)
    rows = []
    for cell in cell_cols:
        active_frames = c_binary.loc[c_binary[cell] == 1.0, "frame"].astype(int)
        cam2_active = [frame_to_cam2[f] for f in active_frames if f in frame_to_cam2]
        rows.append({
            "cell": cell,
            "active_cam2_frames": ",".join(str(f) for f in cam2_active),
            "n_active_frames": len(cam2_active),
        })
    
    result = pd.DataFrame(rows)
    
    return result, skipped


if __name__ == "__main__":
    # File paths - adjust these as needed
    TIMESTAMP_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\cleaning data\timestamp conversion\timestamp_paired.csv"
    C_BINARY_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\Minian_data\scripts\minian\C_binary.csv"
    OUTPUT_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\data_matching\C_binary_active_DLC_frames.csv"
    
    # Check if input files exist
    if not os.path.exists(TIMESTAMP_PATH):
        print(f"Input file not found: {TIMESTAMP_PATH}")
    elif not os.path.exists(C_BINARY_PATH):
        print(f"Input file not found: {C_BINARY_PATH}")
    else:
        # Process the data
        try:
            result, skipped = match_cam0_to_cam2_frames(TIMESTAMP_PATH, C_BINARY_PATH)
            
            # Save results
            result.to_csv(OUTPUT_PATH, index=False)
            
            # Print results
            if skipped:
                print(f"Warning: {skipped} C_binary frames had no matching cam0_frame in timestamps and were skipped.")
            
            print(f"\nDone. {len(result)} cells written to:\n  {OUTPUT_PATH}")
            print(result[["cell", "n_active_frames"]].to_string(index=False))
            print("\nSample active cam2 frames per cell:")
            for _, r in result.iterrows():
                frames_preview = r["active_cam2_frames"].split(",")[:10]
                print(f"  {r['cell']}: {frames_preview}{'...' if r['n_active_frames'] > 10 else ''}")
        except Exception as e:
            print(f"Error processing data: {e}")
