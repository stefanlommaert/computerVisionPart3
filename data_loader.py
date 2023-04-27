import numpy as np
import os
from PIL import Image


IMG_SIZE = (256,256)
IMG_PATH = "./data/train/img"
SEG_PATH = "./data/train/seg"
PREPPED_IMG_PATH = "./prepped_data/train_images/"
PREPPED_SEG_PATH = "./prepped_data/train_segmentation/"

# Create folder to store resized images
if not os.path.exists(PREPPED_IMG_PATH):
    os.makedirs(PREPPED_IMG_PATH)
# Create folder to store resized images
if not os.path.exists(PREPPED_SEG_PATH):
    os.makedirs(PREPPED_SEG_PATH)

for data, prepped_data in [(IMG_PATH, PREPPED_IMG_PATH), (SEG_PATH, PREPPED_SEG_PATH)]:
    # Iterate over all .npy files in train folder
    for filename in os.listdir(data):
        if filename.endswith(".npy"):
            # Load .npy file
            img = np.load(os.path.join(data, filename))

            # Resize image
            img = Image.fromarray(img)
            img = img.resize(IMG_SIZE, resample=Image.NEAREST)
            # Save resized image
            save_path = os.path.join(prepped_data, os.path.splitext(filename)[0] + ".png")
            img.save(save_path)
