import os

import cv2
import numpy as np
import pandas as pd


def hex_to_bgr(hex_color):
    """
    Convert hex color string to BGR tuple.
    
    Input: hex_color - string like "#F0F0F0" or "F0F0F0"
    Output: BGR tuple like (240, 240, 240)
    """
    if isinstance(hex_color, tuple):
        return hex_color  # Already BGR tuple
    if hex_color is None:
        return None
    
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # Return as BGR


def load_dlc_data(dlc_data_path):
    """
    Load DLC multi-header CSV and flatten to single-level column names.

    Inputs:
    - dlc_data_path: path to the DLC CSV file with multi-level headers

    Output:
    - DataFrame with columns like "bodypart_x", "bodypart_y", "bodypart_likelihood"
    """
    dlc_raw = pd.read_csv(dlc_data_path, header=[0, 1, 2], index_col=0)
    dlc_raw.columns = ["_".join(c[1:]).strip() for c in dlc_raw.columns]
    dlc_raw.index = dlc_raw.index.astype(int)
    return dlc_raw


def load_active_frames(active_frames_path):
    """
    Load the per-cell active frame CSV.

    Input:
    - active_frames_path: path to timestamps CSV

    Output:
    - DataFrame with columns: cell, active_cam2_frames, n_active_frames
    """
    return pd.read_csv(active_frames_path)


def get_background_frame(video_path, dlc_raw):
    """
    Return the first frame of a video or a blank canvas sized to DLC coordinates.

    Inputs:
    - video_path: path to a behaviour video (optional)

    Output:
    - background frame as a numpy array (BGR)
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, bg_frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read first frame from: {video_path}")
        return bg_frame

    all_x = dlc_raw[[c for c in dlc_raw.columns if c.endswith("_x")]].values.flatten()
    all_y = dlc_raw[[c for c in dlc_raw.columns if c.endswith("_y")]].values.flatten()
    width = int(np.nanmax(all_x)) + 50
    height = int(np.nanmax(all_y)) + 50
    return np.ones((height, width, 3), dtype=np.uint8) * 245


def get_bodypart_columns(dlc_raw):
    """
    Identify x/y/likelihood columns for plotting.

    Input:
    - dlc_raw: DataFrame with DLC data and columns like "bodypart_x", "bodypart_y", "bodypart_likelihood"

    Output:
    - x_col: name of the x-coordinate column
    - y_col: name of the y-coordinate column
    - lk_col: name of the likelihood column (or None if not found)(optional, but useful for later)
    """
    x_cols = [c for c in dlc_raw.columns if c.endswith("_x")]
    y_cols = [c for c in dlc_raw.columns if c.endswith("_y")]
    if not x_cols or not y_cols:
        raise ValueError("No DLC x/y columns found.")

    lk_cols = [c for c in dlc_raw.columns if c.endswith("_likelihood")]
    lk_col = lk_cols[0] if lk_cols else None
    return x_cols[0], y_cols[0], lk_col


def parse_active_frames(frames_str):
    """
    Parse a comma-separated list of frame indices into integers.

    Input:
    - frames_str: string like "10,15,20" or "10, 15, 20"
    """
    return [int(f) for f in str(frames_str).split(",") if f.strip()]


def add_colorbar_legend(heatmap_img, min_val=0, max_val=255, colormap=cv2.COLORMAP_JET, use_zscore=False, border_color=(128, 64, 0), border_thickness=2):
    """
    Add a colorbar legend to the heatmap image.
    
    Returns a new image with the colorbar appended to the right side.
    border_color: BGR tuple or hex string for border color
    border_thickness: thickness of borders in pixels
    """
    # Convert hex to BGR if needed
    border_color = hex_to_bgr(border_color)
    
    height, width = heatmap_img.shape[:2]
    bar_width = 40
    vertical_padding = 20  # Padding at top and bottom of colorbar
    bar_height = height - (2 * vertical_padding)
    margin = 10
    
    # Create colorbar
    colorbar = np.linspace(255, 0, bar_height).reshape(-1, 1)
    colorbar = np.tile(colorbar, (1, bar_width)).astype(np.uint8)
    colorbar_colored = cv2.applyColorMap(colorbar, colormap)
    
    # Add border to colorbar
    cv2.rectangle(colorbar_colored, (0, 0), (bar_width - 1, bar_height - 1), border_color, border_thickness)
    
    # Create new canvas with space for colorbar and text
    text_width = 50  # Space for labels
    canvas_width = width + bar_width + margin * 2 + text_width
    canvas = np.ones((height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place heatmap with border
    canvas[:, :width] = heatmap_img
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), border_color, border_thickness)
    
    # Place colorbar with vertical padding
    colorbar_x = width + margin
    colorbar_y = vertical_padding
    canvas[colorbar_y:colorbar_y + bar_height, colorbar_x:colorbar_x + bar_width] = colorbar_colored
    
    # Add labels 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)
    
    if use_zscore:
        # Z-score labels - positioned to align with colorbar sections
        labels = [(vertical_padding + 10, "+3σ"), (height // 2, "0σ"), (height - vertical_padding - 10, "-3σ")]
    else:
        # Regular count/intensity labels
        labels = [(vertical_padding + 10, "High"), (height // 2, "Med"), (height - vertical_padding - 10, "Low")]
    
    label_x = colorbar_x + bar_width + 5
    for y_pos, text in labels:
        cv2.putText(canvas, text, (label_x, y_pos + 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return canvas


def build_heatmap(heatmap, blur_sigma, max_value, output_shape, colormap=cv2.COLORMAP_JET, use_zscore=False, background_color=None):
    """
    Normalize and colorize a heatmap on a blank canvas.
    
    colormap options: cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS, 
                      cv2.COLORMAP_TURBO, cv2.COLORMAP_INFERNO, etc.
    use_zscore: if True, normalize using z-score instead of min-max
    background_color: BGR tuple for zero/background values, or None to use colormap default
    """
    if heatmap.max() <= 0:
        return np.zeros(output_shape, dtype=np.uint8)

    heatmap_blur = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)
    
    # Create mask for zero/near-zero values before normalization
    zero_mask = heatmap_blur < 1e-3
    
    if use_zscore:
        # Z-score normalization: (x - mean) / std
        mean = heatmap_blur.mean()
        std = heatmap_blur.std()
        if std > 0:
            heatmap_norm = (heatmap_blur - mean) / std
            # Map to 0-255 range (clip at ±3 standard deviations)
            heatmap_norm = np.clip((heatmap_norm + 3) / 6 * max_value, 0, max_value)
        else:
            heatmap_norm = np.zeros_like(heatmap_blur)
    else:
        # Min-max normalization
        heatmap_norm = np.clip(
            heatmap_blur / (heatmap_blur.max() + 1e-6) * max_value,
            0,
            max_value,
        )
    
    heatmap_norm = heatmap_norm.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
    
    # Apply custom background color to zero values if specified
    if background_color is not None:
        heatmap_colored[zero_mask] = background_color
    
    return heatmap_colored


def compute_place_field(heatmap, blur_sigma, threshold_pct=0.2):
    """
    Compute the place field from a raw activity heatmap.

    The place field is the contiguous region around the peak activity location
    where firing rate exceeds `threshold_pct * peak_value` (default: 20% of peak).

    Inputs:
    - heatmap: 2D float32 array of raw activity counts (same as fed to build_heatmap)
    - blur_sigma: Gaussian blur sigma (same value used for visualization)
    - threshold_pct: fraction of peak activity defining the field boundary (0.0–1.0)

    Returns a dict with:
    - peak_x, peak_y      : pixel coordinates of peak activity
    - centroid_x, centroid_y : centroid of the place field region
    - area_px             : area of the place field in pixels
    - contour             : OpenCV contour of the place field boundary (or None)
    - mask                : binary uint8 mask of the place field region
    Returns None if the heatmap is empty.
    """
    if heatmap.max() <= 0:
        return None

    blurred = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)

    peak_val = float(blurred.max())
    peak_loc = np.unravel_index(np.argmax(blurred), blurred.shape)
    peak_y, peak_x = int(peak_loc[0]), int(peak_loc[1])

    # Binary mask: all pixels above threshold_pct of peak
    threshold = threshold_pct * peak_val
    binary = (blurred >= threshold).astype(np.uint8)

    # Keep only the connected component that contains the peak
    num_labels, labels = cv2.connectedComponents(binary)
    peak_label = labels[peak_y, peak_x]
    field_mask = (labels == peak_label).astype(np.uint8)

    # Centroid of the place field
    moments = cv2.moments(field_mask)
    if moments["m00"] > 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x, centroid_y = peak_x, peak_y

    area_px = int(field_mask.sum())

    # Outer contour of the place field region
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None

    return {
        "peak_x": peak_x,
        "peak_y": peak_y,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area_px": area_px,
        "contour": contour,
        "mask": field_mask,
    }


def annotate_place_field(
    heatmap_img,
    pf_info,
    contour_color=(255, 255, 255),
    peak_color=(0, 0, 255),
    centroid_color=(0, 255, 255),
    contour_thickness=2,
    peak_radius=6,
):
    """
    Draw place field annotation on a heatmap image.

    Draws:
    - A contour outlining the place field boundary
    - A crosshair + circle at the peak activity location
    - A diamond marker at the field centroid

    Inputs:
    - heatmap_img      : BGR image to annotate (annotates a copy)
    - pf_info          : dict returned by compute_place_field
    - contour_color    : BGR color for the boundary contour (default white)
    - peak_color       : BGR color for the peak marker (default red)
    - centroid_color   : BGR color for centroid marker (default cyan)
    - contour_thickness: line thickness for the contour
    - peak_radius      : radius of the circle around the peak

    Returns annotated copy of the image.
    """
    annotated = heatmap_img.copy()

    if pf_info is None:
        return annotated

    h, w = annotated.shape[:2]

    # Draw place field boundary contour
    if pf_info["contour"] is not None:
        cv2.drawContours(annotated, [pf_info["contour"]], -1, contour_color, contour_thickness)

    # Draw peak marker: circle + crosshair
    px, py = pf_info["peak_x"], pf_info["peak_y"]
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(annotated, (px, py), peak_radius, peak_color, 2)
        cv2.line(annotated, (px - peak_radius - 4, py), (px + peak_radius + 4, py), peak_color, 1)
        cv2.line(annotated, (px, py - peak_radius - 4), (px, py + peak_radius + 4), peak_color, 1)

    # Draw centroid marker: diamond
    cx, cy = pf_info["centroid_x"], pf_info["centroid_y"]
    if 0 <= cx < w and 0 <= cy < h:
        cv2.drawMarker(annotated, (cx, cy), centroid_color, cv2.MARKER_DIAMOND, 12, 2)

    return annotated


def plot_cells_and_heatmaps(
    active_df,
    dlc_raw,
    canvas,
    x_col,
    y_col,
    lk_col,
    likelihood_threshold,
    cell_colours,
    default_colours,
    dot_radius,
    dot_thickness,
    heatmap_output_dir,
    heatmap_blur_sigma,
    heatmap_max_value,
    heatmap_colormap=cv2.COLORMAP_JET,
    heatmap_use_zscore=False,
    add_colorbar=True,
    heatmap_background_color=None,
    heatmap_border_color=None,
    heatmap_border_thickness=2,
    compute_place_fields=True,
    place_field_threshold_pct=0.2,
    pf_contour_color=(255, 255, 255),
    pf_peak_color=(0, 0, 255),
    pf_centroid_color=(0, 255, 255),
):
    """
    Plot all cells on the combined canvas and save per-cell heatmaps.

    When compute_place_fields=True, each per-cell heatmap is annotated with:
    - A contour outlining the place field boundary (region above
      place_field_threshold_pct * peak activity)
    - A crosshair/circle at the peak activity pixel
    - A diamond at the field centroid
    Returns a list of place field summary dicts (one per cell).
    """
    os.makedirs(heatmap_output_dir, exist_ok=True)
    place_field_records = []

    for i, (_, row) in enumerate(active_df.iterrows()):
        cell_name = row["cell"]
        active_frames = parse_active_frames(row["active_cam2_frames"])
        colour = cell_colours.get(cell_name, default_colours[i % len(default_colours)])

        heatmap = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.float32)
        plotted = 0

        for frame_idx in active_frames:
            if frame_idx not in dlc_raw.index:
                continue
            x = dlc_raw.at[frame_idx, x_col]
            y = dlc_raw.at[frame_idx, y_col]
            if pd.isna(x) or pd.isna(y):
                continue
            if lk_col and dlc_raw.at[frame_idx, lk_col] < likelihood_threshold:
                continue

            cv2.circle(canvas, (int(x), int(y)), dot_radius, colour, dot_thickness)
            xi, yi = int(x), int(y)
            if 0 <= yi < heatmap.shape[0] and 0 <= xi < heatmap.shape[1]:
                heatmap[yi, xi] += 1.0
            plotted += 1

        print(f"{cell_name}: {plotted}/{len(active_frames)} points plotted")

        heatmap_color = build_heatmap(
            heatmap,
            heatmap_blur_sigma,
            heatmap_max_value,
            canvas.shape,
            colormap=heatmap_colormap,
            use_zscore=heatmap_use_zscore,
            background_color=heatmap_background_color,
        )

        # Compute and annotate place field (before colorbar so coordinates stay correct)
        if compute_place_fields:
            pf_info = compute_place_field(heatmap, heatmap_blur_sigma, place_field_threshold_pct)
            if pf_info is not None:
                heatmap_color = annotate_place_field(
                    heatmap_color,
                    pf_info,
                    contour_color=pf_contour_color,
                    peak_color=pf_peak_color,
                    centroid_color=pf_centroid_color,
                )
                place_field_records.append({
                    "cell": cell_name,
                    "peak_x": pf_info["peak_x"],
                    "peak_y": pf_info["peak_y"],
                    "centroid_x": pf_info["centroid_x"],
                    "centroid_y": pf_info["centroid_y"],
                    "area_px": pf_info["area_px"],
                    "threshold_pct": place_field_threshold_pct,
                })
                print(
                    f"  Place field: peak=({pf_info['peak_x']}, {pf_info['peak_y']}), "
                    f"centroid=({pf_info['centroid_x']}, {pf_info['centroid_y']}), "
                    f"area={pf_info['area_px']} px"
                )
            else:
                print(f"  Place field: no activity")

        # Add colorbar legend if requested
        if add_colorbar:
            heatmap_color = add_colorbar_legend(
                heatmap_color,
                0,
                heatmap_max_value,
                colormap=heatmap_colormap,
                use_zscore=heatmap_use_zscore,
                border_color=heatmap_border_color,
                border_thickness=heatmap_border_thickness,
            )

        heatmap_path = os.path.join(heatmap_output_dir, f"{cell_name}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_color)

    return place_field_records


def draw_legend(canvas, active_df, cell_colours, default_colours, legend_font):
    """
    Draw a legend for each cell on the combined canvas.
    """
    legend_x, legend_y = 10, 20
    for i, (_, row) in enumerate(active_df.iterrows()):
        cell_name = row["cell"]
        colour = cell_colours.get(cell_name, default_colours[i % len(default_colours)])
        cy = legend_y + i * 22
        cv2.circle(canvas, (legend_x + 8, cy), 7, colour, -1)
        cv2.putText(
            canvas,
            cell_name,
            (legend_x + 20, cy + 5),
            legend_font,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


def run_visual_plotting(
    active_frames_path,
    dlc_data_path,
    output_image_path,
    heatmap_output_dir,
    video_path=None,
    cell_colours=None,
    default_colours=None,
    dot_radius=4,
    dot_thickness=-1,
    likelihood_threshold=0.5,
    heatmap_blur_sigma=9,
    heatmap_max_value=255,
    heatmap_colormap=cv2.COLORMAP_JET,
    heatmap_use_zscore=False,
    add_colorbar=True,
    heatmap_background_color=None,
    heatmap_border_color=None,
    heatmap_border_thickness=2,
    compute_place_fields=True,
    place_field_threshold_pct=0.2,
    pf_contour_color=(255, 255, 255),
    pf_peak_color=(0, 0, 255),
    pf_centroid_color=(0, 255, 255),
    place_field_output_csv=None,
):
    """
    Create the combined cell activity plot and per-cell heatmaps.

    When compute_place_fields=True (default), each heatmap image is annotated
    with the detected place field boundary, peak, and centroid. A summary CSV
    is written to place_field_output_csv (defaults to
    <heatmap_output_dir>/place_fields.csv).
    """
    if cell_colours is None:
        cell_colours = {}
    if default_colours is None:
        default_colours = []

    dlc_raw = load_dlc_data(dlc_data_path)
    active_df = load_active_frames(active_frames_path)
    bg_frame = get_background_frame(video_path, dlc_raw)
    canvas = bg_frame.copy()

    x_col, y_col, lk_col = get_bodypart_columns(dlc_raw)

    place_field_records = plot_cells_and_heatmaps(
        active_df,
        dlc_raw,
        canvas,
        x_col,
        y_col,
        lk_col,
        likelihood_threshold,
        cell_colours,
        default_colours,
        dot_radius,
        dot_thickness,
        heatmap_output_dir,
        heatmap_blur_sigma,
        heatmap_max_value,
        heatmap_colormap=heatmap_colormap,
        heatmap_use_zscore=heatmap_use_zscore,
        add_colorbar=add_colorbar,
        heatmap_background_color=heatmap_background_color,
        heatmap_border_color=heatmap_border_color,
        heatmap_border_thickness=heatmap_border_thickness,
        compute_place_fields=compute_place_fields,
        place_field_threshold_pct=place_field_threshold_pct,
        pf_contour_color=pf_contour_color,
        pf_peak_color=pf_peak_color,
        pf_centroid_color=pf_centroid_color,
    )

    # Save place field summary CSV
    if compute_place_fields and place_field_records:
        if place_field_output_csv is None:
            place_field_output_csv = os.path.join(heatmap_output_dir, "place_fields.csv")
        pf_df = pd.DataFrame(place_field_records)
        pf_df.to_csv(place_field_output_csv, index=False)
        print(f"\nSaved place field summary to:\n  {place_field_output_csv}")

    draw_legend(canvas, active_df, cell_colours, default_colours, cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imwrite(output_image_path, canvas)
    print(f"\nSaved plot to:\n  {output_image_path}")

    return output_image_path


if __name__ == "__main__":
    # Resolve paths relative to this script's directory
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Pathing parameters - adjust these as needed
    ACTIVE_FRAMES_PATH = os.path.join(_SCRIPT_DIR, "C_binary_active_DLC_frames.csv")
    DLC_DATA_PATH = os.path.join(_SCRIPT_DIR, "..", "DLC_data", "to csv", "combined_dlc_data.csv")
    OUTPUT_IMAGE_PATH = os.path.join(_SCRIPT_DIR, "cell_activity_plot.png")
    HEATMAP_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "heatmaps")

    # Optional: path to a behaviour video to use as background.
    # Leave as None to use a plain white canvas instead.
    VIDEO_PATH = r"D:\Batch 1\24\24_20190904_T1\behavCam1.avi"
    # VIDEO_PATH = None

    # Colours per cell (BGR)
    """
    (optional)
    CELL_COLOURS = {
        "cell_0_binary": (255, 60, 60),
        "cell_1_binary": (60, 220, 60),
        "cell_2_binary": (60, 60, 255),
        "cell_4_binary": (60, 220, 220),
        ....
    }
    """
    DEFAULT_COLOURS = [
        (255, 160, 48),   # Orange
        (180, 48, 255),   # Purple
        (255, 255, 48),   # Yellow
        (48, 255, 200),   # Cyan
        (255, 100, 180),  # Pink
        (100, 255, 100),  # Light Green
        (255, 140, 0),    # Dark Orange
        (64, 128, 255),   # Light Blue
        (200, 100, 255),  # Lavender
        (255, 200, 100),  # Peach
    ]
    

    # Plotting parameters
    DOT_RADIUS = 4
    DOT_THICKNESS = -1
    LIKELIHOOD_THRESHOLD = 0.5  # DLC confidence filter: only plot points with confidence > this value (0.0-1.0)
    HEATMAP_BLUR_SIGMA = 9 # Blur amount: lower = sharper hotspots, higher = more diffuse (try 2-10)
    HEATMAP_MAX_VALUE = 255

    # Place field parameters
    COMPUTE_PLACE_FIELDS = True   # Set to False to skip place field detection
    PLACE_FIELD_THRESHOLD_PCT = 0.2  # Region above this fraction of peak counts is the place field (0.0–1.0)
    PF_CONTOUR_COLOR = (255, 255, 255)  # BGR color for the field boundary contour
    PF_PEAK_COLOR = (0, 0, 255)         # BGR color for the peak marker (red)
    PF_CENTROID_COLOR = (0, 255, 255)   # BGR color for the centroid marker (cyan)
    PLACE_FIELD_OUTPUT_CSV = None       # None → auto-saves as <HEATMAP_OUTPUT_DIR>/place_fields.csv
    
    # Heatmap visualization options
    HEATMAP_COLORMAP = cv2.COLORMAP_JET  # Options on https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
    HEATMAP_USE_ZSCORE = False  # Set to True for z-score normalization
    ADD_COLORBAR = True  # Set to True to add colorbar legend to heatmaps
    HEATMAP_BACKGROUND_COLOR = hex_to_bgr("#000000")  # Hex or BGR color for zero/background values, None to use colormap default
    # Examples: "#FFFFFF" (white), "#000000" (black), "#F5EBD9" (beige), or BGR tuple like (240, 240, 240)
    HEATMAP_BORDER_COLOR = hex_to_bgr("#804000")  # Hex or BGR for border color around heatmap and colorbar
    HEATMAP_BORDER_THICKNESS = 2  # Border thickness in pixels

    if not os.path.exists(ACTIVE_FRAMES_PATH):
        print(f"Active frames file not found: {ACTIVE_FRAMES_PATH}")
    if not os.path.exists(DLC_DATA_PATH):
        print(f"DLC data file not found: {DLC_DATA_PATH}")

    if VIDEO_PATH and not os.path.exists(VIDEO_PATH):
        print(f"Video not found, using blank canvas: {VIDEO_PATH}")
        VIDEO_PATH = None

    try:
        run_visual_plotting(
            ACTIVE_FRAMES_PATH,
            DLC_DATA_PATH,
            OUTPUT_IMAGE_PATH,
            HEATMAP_OUTPUT_DIR,
            video_path=VIDEO_PATH,
            cell_colours=None, # Optional: specify custom colours per cell in parameters above
            default_colours=DEFAULT_COLOURS,
            dot_radius=DOT_RADIUS,
            dot_thickness=DOT_THICKNESS,
            likelihood_threshold=LIKELIHOOD_THRESHOLD,
            heatmap_blur_sigma=HEATMAP_BLUR_SIGMA,
            heatmap_max_value=HEATMAP_MAX_VALUE,
            heatmap_colormap=HEATMAP_COLORMAP,
            heatmap_use_zscore=HEATMAP_USE_ZSCORE,
            add_colorbar=ADD_COLORBAR,
            heatmap_background_color=HEATMAP_BACKGROUND_COLOR,
            heatmap_border_color=HEATMAP_BORDER_COLOR,
            heatmap_border_thickness=HEATMAP_BORDER_THICKNESS,
            compute_place_fields=COMPUTE_PLACE_FIELDS,
            place_field_threshold_pct=PLACE_FIELD_THRESHOLD_PCT,
            pf_contour_color=PF_CONTOUR_COLOR,
            pf_peak_color=PF_PEAK_COLOR,
            pf_centroid_color=PF_CENTROID_COLOR,
            place_field_output_csv=PLACE_FIELD_OUTPUT_CSV,
        )
    except Exception as exc:
        print(f"Error plotting data: {exc}")

