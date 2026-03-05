'''
use cv2 to draw a rectangular ROI on the frame extracted in frame_extract.py
save its coordinates, for later use when comparing it to DLC data. 
'''

from __future__ import annotations

import argparse
import cv2
from pathlib import Path


def draw_roi(image_path: Path) -> tuple[int, int, int, int]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # load the image
    image = cv2.imread(str(image_path))
    # Check if the image was loaded successfully
    if image is None:
        raise RuntimeError(
            f"Failed to load image: {image_path}. "
            "If this is a PNG/JPG, the file may be corrupt or OpenCV can't decode it."
        )

    # Ensure a visible window is created before ROI selection.
    win = "Draw ROI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, image)
    cv2.waitKey(1)

    roi = cv2.selectROI(win, image, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, roi)

    # Preview the selection until the user presses a key.
    preview = image.copy()
    if w > 0 and h > 0:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(win, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x, y, w, h


def is_point_in_roi(px: float, py: float, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> bool:
    """Check if a point (px, py) is inside the rectangular ROI."""
    return roi_x <= px <= (roi_x + roi_w) and roi_y <= py <= (roi_y + roi_h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a rectangular ROI on an image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).with_name("first_frame.png"),
        help="Path to the image to draw on.",
    )
    args = parser.parse_args()

    x, y, w, h = draw_roi(args.image)
    
    if w == 0 and h == 0:
        print("No ROI selected (ESC pressed)")
        return
    
    print(f"ROI coordinates: x={x}, y={y}, width={w}, height={h}")
    print(f"ROI bounds: top-left=({x},{y}), bottom-right=({x+w},{y+h})")
    print(f"\nTo check if a point is in this ROI, use:")
    print(f"is_point_in_roi(point_x, point_y, {x}, {y}, {w}, {h})")

if __name__ == "__main__":
    main()
