'''
What it should do:

Open an interactive window with a frame of the video and allow the user to draw a rectangular ROI
Save these coordinates, and then use them to extract the frames where the mouse is in the ROI.
Return the list of frames
''' 
from pathlib import Path
import cv2

def extract_frame(folder_path, file_type, out_path, file_name):
    '''
    Goal: Extract the first frame of the first video in the folder and save it 
    as a .png for drawing the ROI later. 

    Parameters :
    - folder_path: str, path to the folder containing the BEHAVIOURAL VIDEOS
    - file_type: str, video file extensions to look for (e.g., {".mp4", ".avi", ".mov", ".mkv"})
    - out_path: Path object, directory where the extracted frame will be saved, must be string
    - file_name: str, name of the output image file (e.g., "first_frame.png")

    Output:
    - .png file of the first frame of the first video in the folder.
    '''

    folder = Path(folder_path)
    
    # sort list for videos for later selection
    videos = sorted([p for p in folder.iterdir() if p.suffix in file_type])

    # error for no videos found
    if not videos:
        print(f"No video files found in {folder_path} with types {file_type}")
        return
    # video you want to use. default is index 0
    video = videos[0]
    
    # open video and read first frame
    cap = cv2.VideoCapture(str(video))
    # read first frame
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"Failed to read video: {video}")
        return
    
    cv2.imwrite(str(out_path / file_name), frame)
    print(f"Saved to {out_path / file_name}")


if __name__ == "__main__":
    # Minimal direct-run example:
    # python .\roi-script.py
    extract_frame(
        folder_path=r"C:\Users\Gebruiker\Desktop\MiPro_2026\POC-Jesse-2026-01-30\videos",
        file_type={".avi"},
        out_path=Path(r"C:\Users\Gebruiker\Desktop\MiPro_2026\martial-madness\detecting ROIs"),
        file_name="first_frame.png",
    )