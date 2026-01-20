import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ------ Parameters -----
gt_img_path = r'/home/tinmar/Desktop/Projects/Datasets/Puzzle/raw/ancre-habitee-edges.jpg'
output_dir_path = r'/home/tinmar/Desktop/Projects/Datasets/Puzzle'


# -----------------------


def skeletonize_image(gt_img):
    # Thresholding
    _, binary_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)

    # 1. Scikit-image's skeletonize expects a boolean image (True where white, False where black)
    # We convert the 0-255 image to boolean.
    bool_img = binary_img > 0

    # 2. Apply Skeletonization
    # This is the core step that thins the lines to 1 pixel
    skeleton = skeletonize(bool_img)

    # 3. Convert back to visible image format
    # The output of skeletonize is boolean. We need to convert it back to uint8 (0-255)
    # to display or save it using standard OpenCV functions.
    skeleton_visual = img_as_ubyte(skeleton)

    return skeleton_visual



if not os.path.exists(gt_img_path):
    raise FileNotFoundError(f"The image path {gt_img_path} does not exist.")
# read image
gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
# Skeletonize
gt_skeleton_img = skeletonize_image(gt_img)
# Save image
output_filename = Path(gt_img_path).stem + ".jpg"
output_filepath = Path(output_dir_path) / output_filename
print(output_filename)
print(output_filepath)
cv2.imwrite(str(output_filepath), gt_skeleton_img)







