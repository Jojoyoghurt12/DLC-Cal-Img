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
    # Always use the full colormap range (255 -> 0) so all colours are visible.
    # In z-score mode the labels map the full bar to +3z .. 0z.
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
        # Positive-only z-score labels: top = +3z, middle = +1.5z, bottom = 0z
        labels = [(vertical_padding + 10, "3"), (height // 2, "1.5"), (height - vertical_padding - 10, "0")]
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
            heatmap_norm = np.clip(heatmap_norm / 3 * max_value, 0, max_value)
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
    - threshold_pct: fraction of peak activity defining the field boundary (0.0â€“1.0)

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


def run_density_heatmap(
    dlc_data_path,
    output_path,
    video_path=None,
    likelihood_threshold=0.6,
    blur_sigma=15,
    max_value=255,
    colormap=cv2.COLORMAP_JET,
    use_zscore=False,
    add_colorbar=True,
    background_color=None,
    border_color=None,
    border_thickness=2,
    compute_place_fields=True,
    place_field_threshold_pct=0.2,
    pf_contour_color=(255, 255, 255),
    pf_peak_color=(0, 0, 255),
    pf_centroid_color=(0, 255, 255),
):
    """
    Build a density (dwell-time) heatmap from all DLC position data.

    Every frame where the tracked point passes the likelihood threshold counts
    +1 at that pixel. The accumulated counts are blurred and colourised to
    show where the mouse spent the most time.

    Inputs:
    - dlc_data_path       : path to combined_dlc_data.csv (multi-header DLC CSV)
    - output_path         : where to save the heatmap PNG
    - video_path          : optional behaviour video for background; None â†’ blank canvas
    - likelihood_threshold: minimum DLC confidence to include a frame (0.0â€“1.0)
    - blur_sigma          : Gaussian blur radius; higher = more diffuse blobs
    - max_value           : ceiling for the normalised colour range (usually 255)
    - colormap            : OpenCV colourmap constant (e.g. cv2.COLORMAP_JET)
    - use_zscore          : normalise with z-score instead of min-max
    - add_colorbar        : append a colourbar legend to the right side
    - background_color    : BGR tuple / hex for zero-value pixels; None = colourmap default
    - border_color        : BGR tuple / hex for image border; None = no border
    - border_thickness    : border width in pixels
    - compute_place_fields: annotate peak, centroid, and field boundary on the heatmap
    - place_field_threshold_pct: fraction of peak that defines the field boundary (0.0â€“1.0)
    - pf_contour_color    : BGR colour for the boundary contour
    - pf_peak_color       : BGR colour for the peak marker
    - pf_centroid_color   : BGR colour for the centroid marker

    Returns the saved output path.
    """
    dlc_raw = load_dlc_data(dlc_data_path)
    bg_frame = get_background_frame(video_path, dlc_raw)
    x_col, y_col, lk_col = get_bodypart_columns(dlc_raw)

    heatmap = np.zeros((bg_frame.shape[0], bg_frame.shape[1]), dtype=np.float32)
    plotted = 0
    total = len(dlc_raw)

    for idx, row in dlc_raw.iterrows():
        x, y = row[x_col], row[y_col]
        if pd.isna(x) or pd.isna(y):
            continue
        if lk_col and row[lk_col] < likelihood_threshold:
            continue
        xi, yi = int(x), int(y)
        if 0 <= yi < heatmap.shape[0] and 0 <= xi < heatmap.shape[1]:
            heatmap[yi, xi] += 1.0
            plotted += 1

    print(f"Plotted {plotted}/{total} frames (likelihood >= {likelihood_threshold})")

    heatmap_color = build_heatmap(
        heatmap,
        blur_sigma,
        max_value,
        bg_frame.shape,
        colormap=colormap,
        use_zscore=use_zscore,
        background_color=background_color,
    )

    if compute_place_fields:
        pf_info = compute_place_field(heatmap, blur_sigma, place_field_threshold_pct)
        if pf_info:
            heatmap_color = annotate_place_field(
                heatmap_color,
                pf_info,
                contour_color=pf_contour_color,
                peak_color=pf_peak_color,
                centroid_color=pf_centroid_color,
            )
            print(
                f"Peak:     ({pf_info['peak_x']}, {pf_info['peak_y']})\n"
                f"Centroid: ({pf_info['centroid_x']}, {pf_info['centroid_y']})\n"
                f"Field area: {pf_info['area_px']} px"
            )
        else:
            print("No activity detected â€” heatmap is empty.")

    if add_colorbar:
        heatmap_color = add_colorbar_legend(
            heatmap_color,
            0,
            max_value,
            colormap=colormap,
            use_zscore=use_zscore,
            border_color=border_color,
            border_thickness=border_thickness,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, heatmap_color)
    print(f"\nSaved heatmap to:\n  {output_path}")
    return output_path


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Input / output paths 
    DLC_DATA_PATH  = os.path.join(_SCRIPT_DIR, "combined_dlc_data.csv")
    OUTPUT_PATH    = os.path.join(_SCRIPT_DIR, "density_heatmap.png")

    # Optional: path to a behaviour video to use as background.
    # Set to None to use a plain white canvas sized to the DLC coordinates.
    VIDEO_PATH = None
    # VIDEO_PATH = r"D:\Batch 1\24\24_20190904_T1\behavCam1.avi"

    # Parameters
    LIKELIHOOD_THRESHOLD = 0.6   # Discard frames where DLC confidence is below this (0.0–1.0)

    # Heatmap
    BLUR_SIGMA   = 15            # Blur radius: lower = sharper hotspots, higher = more diffuse
    MAX_VALUE    = 255
    COLORMAP     = cv2.COLORMAP_JET   # See https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
    USE_ZSCORE   = True          # True = z-score normalisation instead of min-max
    ADD_COLORBAR = True
    BACKGROUND_COLOR  = hex_to_bgr("#000000")   # Colour for pixels with zero dwell time
    BORDER_COLOR      = hex_to_bgr("#804000")
    BORDER_THICKNESS  = 2

    # Annotation
    COMPUTE_PLACE_FIELDS     = True
    PLACE_FIELD_THRESHOLD_PCT = 0.2   # Region above 20 % of peak counts = place field
    PF_CONTOUR_COLOR = (255, 255, 255)   # White boundary contour
    PF_PEAK_COLOR    = (0,   0,   255)   # Red peak marker
    PF_CENTROID_COLOR = (0, 255, 255)    # Cyan centroid marker

    # Validation
    if not os.path.exists(DLC_DATA_PATH):
        print(f"DLC data file not found: {DLC_DATA_PATH}")
    else:
        if VIDEO_PATH and not os.path.exists(VIDEO_PATH):
            print(f"Video not found, using blank canvas: {VIDEO_PATH}")
            VIDEO_PATH = None

        try:
            run_density_heatmap(
                DLC_DATA_PATH,
                OUTPUT_PATH,
                video_path=VIDEO_PATH,
                likelihood_threshold=LIKELIHOOD_THRESHOLD,
                blur_sigma=BLUR_SIGMA,
                max_value=MAX_VALUE,
                colormap=COLORMAP,
                use_zscore=USE_ZSCORE,
                add_colorbar=ADD_COLORBAR,
                background_color=BACKGROUND_COLOR,
                border_color=BORDER_COLOR,
                border_thickness=BORDER_THICKNESS,
                compute_place_fields=COMPUTE_PLACE_FIELDS,
                place_field_threshold_pct=PLACE_FIELD_THRESHOLD_PCT,
                pf_contour_color=PF_CONTOUR_COLOR,
                pf_peak_color=PF_PEAK_COLOR,
                pf_centroid_color=PF_CENTROID_COLOR,
            )
        except Exception as exc:
            print(f"Error building heatmap: {exc}")
            raise

