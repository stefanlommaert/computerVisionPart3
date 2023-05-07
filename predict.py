from keras_segmentation.predict import predict, predict_multiple

class_names = ['bg', "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
          (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), 
          (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
          (192, 0, 0), (0,192,0)]

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
import os
import numpy as np
from PIL import Image
import pickle
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from ownmodel import dice_loss, focal_tversky_loss

def predict(model, X):
    # resize the input image to the network input size
    sizes = [img.shape[:2] for img in X]
    print(sizes)
    X = np.array([cv2.resize(img, (256, 256)) for img in X])
    # make predictions
    X = model.predict(X)
    print(X.shape)
    # # print the class probabilities for each pixel
    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         for k in range(len(X[i][j])):
    #             print(X[i][j][k])
    # predict the class probabilities for each pixel
    X = np.argmax(X, axis=-1)
    print(X.shape)
    return X
    # for i in range(len(X)):
    #     print(X[i].shape)
    # # back to the original size
    # return np.array([cv2.resize(img, size) for img, size in zip(X, sizes)])

predict( 
	checkpoints_path="checkpoints3/resnet_unet_1.54", 
	inp="prepped_data/val_images/augmented_0_train_0.png", 
	out_fname="outputfinal.png",
	class_names=class_names,
    colors= colors,
	prediction_height=128,
    prediction_width=128
)


# predict_multiple( 
# 	checkpoints_path="checkpoints3/resnet_unet_1.54", 
# 	inp_dir="prepped_data/test_images/", 
# 	out_dir="predicted_images",
#     colors=colors,
#     class_names=class_names,
#     prediction_height=128,
#     prediction_width=128
# )

def count_pixels(arr):
    arr = arr.flatten()
    count = [0] * 21
    for i in range(len(arr)):
        count[arr[i]] += 1
    return count

# load test images from working/test_imagess
test_X = np.array([np.array(Image.open(f"./working/test_imagess/{i}")) for i in os.listdir("./working/test_imagess/")])

# load the model
model = load_model("./working/model.h5", custom_objects={'dice_loss': dice_loss, 'focal_tversky_loss': focal_tversky_loss})

model.load_weights("./model-01-dl.h5")

# predict the segmentation of the test images
test_y_seg = predict(model, test_X[:10])

# # plot the segmentation mask for the first image
plt.imshow(test_y_seg[0], vmin=0, vmax=20)
plt.show()

# # save the segmentation images to working/predictions
for i in range(len(test_y_seg)):
    img = Image.fromarray(test_y_seg[i].astype(np.uint8))
    img.save(f"./working/predictions/{i}.png")
    print(count_pixels(test_y_seg[i].flatten()))
    img = Image.fromarray(test_X[i].astype(np.uint8))
    img.save(f"./working/predictions/{i}_X.png")
