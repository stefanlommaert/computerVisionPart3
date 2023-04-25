import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import to_categorical  # One-hot encoding

def fit(self, X, Y):
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

def predict(self, X):
    return np.argmax(self.model.predict(X), axis=-1)	