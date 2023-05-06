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
