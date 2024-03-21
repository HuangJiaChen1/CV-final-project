import cv2
import os
import numpy as np

# Directory paths
dir_lk = "frame_lk"
dir_dl = "frame_dl"
dir_vis = "frame_vis"

def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])
# Create the frame_vis directory if it doesn't exist
if not os.path.exists(dir_vis):
    os.makedirs(dir_vis)
files_lk = sorted(os.listdir(dir_lk), key=extract_number)

# Assuming both directories contain the same number of files and corresponding files have the same names
for filename in files_lk:
    if filename.endswith(".jpg"):
        # Construct the path to the current image in both directories
        path_lk = os.path.join(dir_lk, filename)
        path_dl = os.path.join(dir_dl, filename)
        print(path_lk)
        # Load the images
        image_lk = cv2.imread(path_lk)
        image_dl = cv2.imread(path_dl)

        if image_lk is not None and image_dl is not None:
            overlaid_image = cv2.addWeighted(image_lk, 0.5, image_dl, 0.5, 0)
            save_path = os.path.join(dir_vis, filename)
            cv2.imwrite(save_path, overlaid_image)
        else:
            print(f"Skipping {filename}, as one of the images could not be loaded.")

print("Processing complete.")

files_vis = sorted(os.listdir(dir_vis), key=extract_number)
for filename in files_vis:
    if filename.endswith(".jpg"):
        path = os.path.join(dir_vis, filename)
        image = cv2.imread(path)
        cv2.imshow('image',image)
        cv2.waitKey(0)