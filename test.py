from matplotlib import pyplot as plt
import numpy as np

# img_array = np.load('data/train/seg/train_22.npy')


# print(img_array)

# plt.imshow(img_array1, cmap='gray')
# plt.show()


# plt.show()



from PIL import Image

def count_colors(image_path):
    with Image.open(image_path) as img:
        colors = img.getcolors(maxcolors=256**3)
        return len(colors)

print(count_colors('prepped_data/train_segmentation/augmented1_train_22.png'))