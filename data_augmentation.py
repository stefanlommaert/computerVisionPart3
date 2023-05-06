import albumentations as A
import cv2
from PIL import Image
import numpy as np
import os

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3)
    ], p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)
    ], p=0.5),
    A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.RandomResizedCrop(height = 256, width = 256, scale = (0.70, 1), ratio = (0.95, 1.05), p = 0.5)
])


PREPPED_IMG_PATH = "./prepped_data/train_images/"
PREPPED_SEG_PATH = "./prepped_data/train_segmentation/"


# Iterate over all .npy files in train folder
for filename in os.listdir(PREPPED_IMG_PATH):
    if not filename.startswith("augmented"):
        image = cv2.imread(os.path.join(PREPPED_IMG_PATH, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(PREPPED_SEG_PATH, filename))

        for i in range(10):
            transformed = transform(image=image, mask=mask)
            transformed_image = Image.fromarray(transformed['image'])
            transformed_mask = Image.fromarray(transformed['mask'])

            save_path = os.path.join(PREPPED_IMG_PATH, f"augmented_{i}_"+os.path.splitext(filename)[0] + ".png")
            transformed_image.save(save_path)
            save_path = os.path.join(PREPPED_SEG_PATH, f"augmented_{i}_"+os.path.splitext(filename)[0] + ".png")
            transformed_mask.save(save_path)
