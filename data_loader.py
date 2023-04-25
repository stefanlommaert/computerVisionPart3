import numpy as np
import os
from PIL import Image

# Create folder to store resized images
if not os.path.exists("./prepped_data/train_images/"):
    os.makedirs("./prepped_data/train_images/")
# Create folder to store resized images
if not os.path.exists("./prepped_data/train_segmentation/"):
    os.makedirs("./prepped_data/train_segmentation/")

# Iterate over all .npy files in train_images folder
for filename in os.listdir("./data/train/img"):
    if filename.endswith(".npy"):
        # Load .npy file
        img = np.load(os.path.join("./data/train/img", filename))

        # Resize image
        img = Image.fromarray(img)
        img = img.resize((416, 416), resample=Image.BILINEAR)
        # Save resized image
        save_path = os.path.join("./prepped_data/train_images/", os.path.splitext(filename)[0] + ".png")
        img.save(save_path)
        
# Iterate over all .npy files in train_images folder
for filename in os.listdir("./data/train/seg"):
    if filename.endswith(".npy"):
        # Load .npy file
        img = np.load(os.path.join("./data/train/seg", filename))

        # Resize image
        img = Image.fromarray(img)
        img = img.resize((416, 416), resample=Image.BILINEAR)

        # Save resized image
        save_path = os.path.join("./prepped_data/train_segmentation/", os.path.splitext(filename)[0] + ".png")
        img.save(save_path)