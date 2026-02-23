import os

import cv2
import numpy as np
import pandas as pd


def load_dlc_data(dlc_data_path):
    """
    Load DLC multi-header CSV and flatten to single-level column names.
    """
    dlc_raw = pd.read_csv(dlc_data_path, header=[0, 1, 2], index_col=0)
    dlc_raw.columns = ["_".join(c[1:]).strip() for c in dlc_raw.columns]
    dlc_raw.index = dlc_raw.index.astype(int)
    return dlc_raw


def load_active_frames(active_frames_path):
    """
    Load the per-cell active frame CSV.
    """
    return pd.read_csv(active_frames_path)


def get_background_frame(video_path, dlc_raw):
    """
    Return the first frame of a video or a blank canvas sized to DLC coordinates.
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
    """
    return [int(f) for f in str(frames_str).split(",") if f.strip()]


def build_heatmap(heatmap, blur_sigma, max_value, output_shape):
    """
    Normalize and colorize a heatmap on a blank canvas.
    """
    if heatmap.max() <= 0:
        return np.zeros(output_shape, dtype=np.uint8)

    heatmap_blur = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)
    heatmap_norm = np.clip(
        heatmap_blur / (heatmap_blur.max() + 1e-6) * max_value,
        0,
        max_value,
    ).astype(np.uint8)
    return cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)


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
):
    """
    Plot all cells on the combined canvas and save per-cell heatmaps.
    """
    os.makedirs(heatmap_output_dir, exist_ok=True)

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
        )

        heatmap_path = os.path.join(heatmap_output_dir, f"{cell_name}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_color)


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
):
    """
    Create the combined cell activity plot and per-cell heatmaps.
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

    plot_cells_and_heatmaps(
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
    )

    draw_legend(canvas, active_df, cell_colours, default_colours, cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imwrite(output_image_path, canvas)
    print(f"\nSaved plot to:\n  {output_image_path}")

    return output_image_path


if __name__ == "__main__":
    # Parameters - adjust these as needed
    ACTIVE_FRAMES_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\data_matching\C_binary_active_DLC_frames.csv"
    DLC_DATA_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\DLC_data\to csv\combined_dlc_data.csv"
    OUTPUT_IMAGE_PATH = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\data_matching\cell_activity_plot.png"
    HEATMAP_OUTPUT_DIR = r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\data_matching\heatmaps"

    # Optional: path to a behaviour video to use as background.
    # Leave as None to use a plain white canvas instead.
    VIDEO_PATH = r"D:\Batch 1\24\24_20190904_T1\behavCam1.avi"
    # VIDEO_PATH = None

    # Colours per cell (BGR)
    CELL_COLOURS = {
        "cell_0_binary": (255, 60, 60),
        "cell_1_binary": (60, 220, 60),
        "cell_2_binary": (60, 60, 255),
        "cell_4_binary": (60, 220, 220),
    }
    DEFAULT_COLOURS = [
        (255, 160, 48),
        (180, 48, 255),
        (255, 255, 48),
        (48, 255, 200),
    ]

    DOT_RADIUS = 4
    DOT_THICKNESS = -1
    LIKELIHOOD_THRESHOLD = 0.5
    HEATMAP_BLUR_SIGMA = 9
    HEATMAP_MAX_VALUE = 255

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
            cell_colours=CELL_COLOURS,
            default_colours=DEFAULT_COLOURS,
            dot_radius=DOT_RADIUS,
            dot_thickness=DOT_THICKNESS,
            likelihood_threshold=LIKELIHOOD_THRESHOLD,
            heatmap_blur_sigma=HEATMAP_BLUR_SIGMA,
            heatmap_max_value=HEATMAP_MAX_VALUE,
        )
    except Exception as exc:
        print(f"Error plotting data: {exc}")

