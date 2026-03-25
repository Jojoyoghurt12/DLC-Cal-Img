import os
import re

def rename_videos_in_folder(folder_path):
    """
    Renames all video files for chronological ordering later

    inputs : 
    folder_path: str, path to the folder containing the video files.
        example : "D:/Manual T-maze recordings/47_20190426/"
        would change all videos for april 26th 
        

    outputs : 
    Renames all video files in the folder to a standardized format for easier chronological ordering.
        example format name before : /47_yyyymmdd/behavCam1.avi
        example format name after  : /47_yyyymmdd/mm-dd-behavCam1.avi

    This way you can sort the .h5 output files from DLC alphabetically and
    they will be in chronological order for easier combination later.
    """

    # extract date from folder name
    folder_name = os.path.basename(folder_path)
    # find pattern of 4 digits (year), followed by 2 digits (month), followed by 2 digits (day)
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})\s+Tmaze', folder_name)

    # error contrl
    if not date_match:  
        print(f"Date not found in folder name: {folder_name}. Skipping renaming.")
        return
    
    # extract month (second grouping of re) and day (third grouping of re)
    mm = date_match.group(2) 
    dd = date_match.group(3)

    # rename all .avi files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.avi'):
            # constitute new filename
            new_filename = f"{mm}-{dd}-{filename}"
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            # get list of all renamed files for confirmation
            print(f"Renamed {filename} to {new_filename}")

if __name__ == "__main__":
    # specify folder for transformation.
    folder_path = "D:\\Manual T-maze recordings\\47_20190426 Tmaze"
    rename_videos_in_folder(folder_path)
    print("Renaming completed. All video files in the folder have been renamed for chronological ordering.")