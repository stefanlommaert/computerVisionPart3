# from PIL import Image

# def count_colors(image_path):
#     with Image.open(image_path) as img:
#         colors = img.getcolors(maxcolors=256**3)
#         return len(colors)

# print(count_colors('output1.png'))



# import csv

# class_count = {}

# with open('data/train/train_set.csv', 'r') as f:
#     reader = csv.DictReader(f)
#     i = 0
#     for row in reader:
#         i+=1
#         if i<2:
#             for key, value in row.items():
#                 if key != 'Id':
#                     class_count[key] = class_count.get(key, 0) + int(value)

# for key, value in class_count.items():
#     print(f'{key}: {value}')



from PIL import Image

def get_label_counts(image_path, label_dict):
    """
    Returns the total amount of pixels for each label in the given image,
    where the label colors are defined in the given dictionary.
    """
    # Load the image
    with Image.open(image_path) as img:
        # Convert to RGB if it's in a different mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Get the pixel data
        pixels = img.load()
        width, height = img.size
        
        # Initialize the label counts to 0
        label_counts = {label: 0 for label in label_dict}
        
        # Iterate over each pixel
        for x in range(width):
            for y in range(height):
                # Get the pixel color tuple
                color = pixels[x, y]
                
                # Check if it matches any of the label colors
                for label, label_color in label_dict.items():
                    if color == label_color:
                        # Increment the count for that label
                        label_counts[label] += 1
                        
        return label_counts
    

from PIL import Image

from PIL import Image
from collections import Counter
import numpy as np

def get_label_percentage(png_file, label_dict):
    # Load the PNG image file
    img = Image.open(png_file)

    # Get the pixels as a flattened list of RGB tuples
    pixels = img.getdata()

    # Create a Counter object to count the occurrence of each color
    color_counts = Counter(pixels)

    # Calculate the total number of pixels in the image
    total_pixels = len(pixels)

    # Calculate the percentage of pixels per label
    label_percentages = {}
    for label, color in label_dict.items():
        if color in color_counts:
            label_pixels = color_counts[color]
            percentage = (label_pixels / total_pixels) * 100
            label_percentages[label] = percentage

    # Get the top 5 labels with the highest percentage of pixels
    top_labels = sorted(label_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_labels



def get_label_array(png_file, label_dict):
    # Load the PNG image file
    img = Image.open(png_file)

    # Get the pixels as a flattened list of RGB tuples
    pixels = img.getdata()

    # Create a Counter object to count the occurrence of each color
    color_counts = Counter(pixels)

    # Calculate the total number of pixels in the image
    total_pixels = len(pixels)

    # Calculate the percentage of pixels per label (excluding background)
    label_percentages = {}
    for label, color in label_dict.items():
        if color in color_counts and label != "background":
            label_pixels = color_counts[color]
            percentage = (label_pixels / total_pixels) * 100
            label_percentages[label] = percentage

    # Create an array of 1s and 0s based on the percentage of pixels per label
    label_array = []
    for label in label_dict.keys():
        if label == "background":
            continue
        if label in label_percentages and label_percentages[label] > 10:
            label_array.append(1)
        else:
            label_array.append(0)

    return np.array(label_array)



color_dict = {
    "background": (0, 0, 0),
    "aeroplane": (1, 1, 1),
    "bicycle": (2, 2, 2),
    "bird": (3, 3, 3),
    "boat": (4, 4, 4),
    "bottle": (5, 5, 5),
    "bus": (6, 6, 6),
    "car": (7, 7, 7),
    "cat": (8, 8, 8),
    "chair": (9, 9, 9),
    "cow": (10, 10, 10),
    "diningtable": (11, 11, 11),
    "dog": (12, 12, 12),
    "horse": (13, 13, 13),
    "motorbike": (14, 14, 14),
    "person": (15, 15, 15),
    "pottedplant": (16, 16, 16),
    "sheep": (17, 17, 17),
    "sofa": (18, 18, 18),
    "train": (19, 19, 19),
    "tvmonitor": (20, 20, 20)
}




# print(get_label_percentage("outputfinall.png", color_dict))




# import numpy as np

# def print_npy_file(filename):
#     arr = np.load(filename)
    
#     return [np.random.choice(np.arange(20 + 1), size=X_.shape[:2]) for X_ in [arr]]
     


# print(len(print_npy_file("data/train/img/train_0.npy")[0]))


import numpy as np
from PIL import Image

def png_to_nparray(filename):
    with Image.open(filename) as img:
        # Convert the image to a NumPy array
        arr = np.array(img)

        # Extract the R channel from the array
        r_arr = arr[:, :, 0]

        return r_arr
    
print(png_to_nparray("outputfinall.png")[64])

