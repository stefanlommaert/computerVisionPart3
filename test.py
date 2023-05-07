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



color_dict = {
    "background": (0, 0, 0),
    "aeroplane": (255, 0, 0),
    "bicycle": (0, 255, 0),
    "bird": (0, 0, 255),
    "boat": (255, 255, 0),
    "bottle": (255, 0, 255),
    "bus": (0, 255, 255),
    "car": (128, 0, 0),
    "cat": (0, 128, 0),
    "chair": (0, 0, 128),
    "cow": (128, 128, 0),
    "diningtable": (128, 0, 128),
    "dog": (0, 128, 128),
    "horse": (64, 0, 0),
    "motorbike": (0, 64, 0),
    "person": (0, 0, 64),
    "pottedplant": (64, 64, 0),
    "sheep": (64, 0, 64),
    "sofa": (0, 64, 64),
    "train": (192, 0, 0),
    "tvmonitor": (0, 192, 0)
}



print(get_label_percentage("predicted_images/test_7.png", color_dict))