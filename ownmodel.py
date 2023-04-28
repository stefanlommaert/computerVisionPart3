import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import to_categorical  # One-hot encoding
from PIL import Image
import os
import pickle

def fit(X, Y):
    # Convert the class labels to one-hot encoded vectors
    Y = to_categorical(Y, num_classes=21)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(21, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, Y, epochs=10, batch_size=4)
    return model

# read training data and train the model from prepped_data/train_images and prepped_data/train_segmentation
train_X = np.array([np.array(Image.open(f"./working/train_X/{i}")) for i in os.listdir("./working/train_X/")])
train_y_seg = np.array([np.array(Image.open(f"./working/train_y_seg/{i}")) for i in os.listdir("./working/train_y_seg/")])

model = fit(train_X, train_y_seg)

# save the model
pickle.dump(model, open('model_own.pkl','wb'))



