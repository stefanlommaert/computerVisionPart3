import os
import numpy as np
from PIL import Image
import pickle

def predict(model, X):
    return np.argmax(model.predict(X), axis=-1)

# load test images from working/test_imagess
test_X = np.array([np.array(Image.open(f"./working/test_imagess/{i}")) for i in os.listdir("./working/test_imagess/")])

# load the model
model = pickle.load(open('model_own.pkl','rb'))

# predict the segmentation of the test images
test_y_seg = predict(model, test_X[:10])

# save the segmentation images to working/predictions
for i in range(len(test_y_seg)):
    img = Image.fromarray(test_y_seg[i].astype(np.uint8))
    img.save(f"./working/predictions/{i}.png")
