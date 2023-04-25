import albumentations as A
import cv2
from PIL import Image
import numpy as np
import os

transform1 = A.Compose([
    A.HorizontalFlip(True),
])

transform2 = A.Compose([
    A.RandomBrightnessContrast(p=1)
])


PREPPED_IMG_PATH = "./prepped_data/train_images/"
PREPPED_SEG_PATH = "./prepped_data/train_segmentation/"


# Iterate over all .npy files in train folder
for filename in os.listdir(PREPPED_IMG_PATH):
    image = cv2.imread(os.path.join(PREPPED_IMG_PATH, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(PREPPED_SEG_PATH, filename))

    transformed = transform1(image=image, mask=mask)
    transformed_image = Image.fromarray(transformed['image'])
    transformed_mask = Image.fromarray(transformed['mask'])

    save_path = os.path.join(PREPPED_IMG_PATH, "augmented1_"+os.path.splitext(filename)[0] + ".png")
    transformed_image.save(save_path)
    save_path = os.path.join(PREPPED_SEG_PATH, "augmented1_"+os.path.splitext(filename)[0] + ".png")
    transformed_mask.save(save_path)

for filename in os.listdir(PREPPED_IMG_PATH):
    image = cv2.imread(os.path.join(PREPPED_IMG_PATH, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(PREPPED_SEG_PATH, filename))

    transformed = transform2(image=image, mask=mask)
    transformed_image = Image.fromarray(transformed['image'])
    transformed_mask = Image.fromarray(transformed['mask'])

    save_path = os.path.join(PREPPED_IMG_PATH, "augmented2_"+os.path.splitext(filename)[0] + ".png")
    transformed_image.save(save_path)
    save_path = os.path.join(PREPPED_SEG_PATH, "augmented2_"+os.path.splitext(filename)[0] + ".png")
    transformed_mask.save(save_path)
