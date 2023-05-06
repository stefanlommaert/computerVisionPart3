from keras.layers import concatenate
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Conv2DTranspose
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical  # One-hot encoding
from PIL import Image
import os
import pickle
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

@keras.utils.register_keras_serializable(package='Custom', name='dice_loss')
def dice_loss(y_true, y_pred):
    num_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(K.cast(y_true[..., 0], 'int32'), num_classes)
    y_true = K.permute_dimensions(y_true, (0, 3, 1, 2))
    y_pred = K.permute_dimensions(y_pred, (0, 3, 1, 2))
    dice_coefs = []
    for c in range(num_classes):
        y_true_c = y_true[:, c]
        y_pred_c = y_pred[:, c]
        numerator = 2 * K.sum(y_true_c * y_pred_c)
        denominator = K.sum(y_true_c + y_pred_c)
        dice_coefs.append(numerator / (denominator + K.epsilon()))
    dice_coefs = K.stack(dice_coefs)
    return 1 - K.mean(dice_coefs)

@keras.utils.register_keras_serializable(package='Custom', name='focal_tversky_loss')
def focal_tversky_loss(y_true, y_pred, class_weights=None):
    # define the focusing parameter
    gamma = 2
    # define the Tversky index parameters
    alpha = 0.7
    beta = 0.3
    # cast y_true and y_pred to float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # calculate the true positives, false positives, and false negatives for each class
    tp = K.sum(y_true * y_pred, axis=[0, 1, 2])
    fp = K.sum((1 - y_true) * y_pred, axis=[0, 1, 2])
    fn = K.sum(y_true * (1 - y_pred), axis=[0, 1, 2])
    # calculate the weighted Tversky index for each class
    class_weights = [1] * 21
    class_weights[0] = 0.5
    class_weights[15] = 0.7
    class_weights = K.constant(class_weights) if class_weights is not None else 1.0
    tversky_index = (tp) / (tp + alpha * fp + beta * fn )
    weighted_tversky_index = class_weights * tversky_index
    # calculate the mean focal Tversky loss over all classes
    focal_tversky_loss = K.mean(K.pow((1 - weighted_tversky_index), 1 / gamma))
    return focal_tversky_loss

@keras.utils.register_keras_serializable(package='Custom', name='iou')
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

@keras.utils.register_keras_serializable(package='Custom', name='mean_iou')
def mean_iou(y_true, y_pred):
    num_classes = K.int_shape(y_pred)[-1]
    iou_scores = []
    for i in range(num_classes):
        true_mask = y_true[...,i]
        pred_mask = y_pred[...,i]
        intersection = K.sum(true_mask * pred_mask, axis=[1,2])
        union = K.sum(true_mask + pred_mask, axis=[1,2]) - intersection
        iou = (intersection + K.epsilon()) / (union + K.epsilon())
        iou_scores.append(iou)
    mean_iou = K.mean(K.stack(iou_scores), axis=0)
    return mean_iou

def data_generator(X, Y, batch_size):
    while True:
        # Shuffle the data
        p = np.random.permutation(len(X))
        X, Y = X[p], Y[p]

        # Create batches
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]

            # normalize the data
            x_batch = x_batch / 255.0

            # Convert the class labels to one-hot encoded vectors
            y_batch = to_categorical(y_batch, num_classes=21)

            yield (x_batch, y_batch)

def build():

    # # Encoder Path
    # inputs = Input(shape=(256, 256, 3))
    # conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    # # conv1 = BatchNormalization()(conv1)
    # conv1 = Activation('relu')(conv1)
    # conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    # # conv1 = BatchNormalization()(conv1)
    # conv1 = Activation('relu')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    # # conv2 = BatchNormalization()(conv2)
    # conv2 = Activation('relu')(conv2)
    # conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    # # conv2 = BatchNormalization()(conv2)
    # conv2 = Activation('relu')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    # # conv3 = BatchNormalization()(conv3)
    # conv3 = Activation('relu')(conv3)
    # conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    # # conv3 = BatchNormalization()(conv3)
    # conv3 = Activation('relu')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    # # conv4 = BatchNormalization()(conv4)
    # conv4 = Activation('relu')(conv4)
    # conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    # # conv4 = BatchNormalization()(conv4)
    # conv4 = Activation('relu')(conv4)


    # # Decoder Path with Dropout
    # up5 = UpSampling2D(size=(2, 2))(conv4)
    # up5 = concatenate([up5, conv3], axis=-1)
    # conv5 = Conv2D(128, (3, 3), padding='same')(up5)
    # # conv5 = BatchNormalization()(conv5)
    # conv5 = Activation('relu')(conv5)
    # conv5 = Conv2D(128, (3, 3), padding='same')(conv5)
    # # conv5 = BatchNormalization()(conv5)
    # conv5 = Activation('relu')(conv5)
    # conv5 = Dropout(0.2)(conv5)

    # up6 = UpSampling2D(size=(2, 2))(conv5)
    # up6 = concatenate([up6, conv2], axis=-1)
    # conv6 = Conv2D(64, (3, 3), padding='same')(up6)
    # # conv6 = BatchNormalization()(conv6)
    # conv6 = Activation('relu')(conv6)
    # conv6 = Conv2D(64, (3, 3), padding='same')(conv6)
    # # conv6 = BatchNormalization()(conv6)
    # conv6 = Activation('relu')(conv6)
    # conv6 = Dropout(0.2)(conv6)

    # up7 = UpSampling2D(size=(2, 2))(conv6)
    # up7 = concatenate([up7, conv1], axis=-1)
    # conv7 = Conv2D(32, (3, 3), padding='same')(up7)
    # # conv7 = BatchNormalization()(conv7)
    # conv7 = Activation('relu')(conv7)
    # conv7 = Conv2D(32, (3, 3), padding='same')(conv7)
    # # conv7 = BatchNormalization()(conv7)
    # conv7 = Activation('relu')(conv7)
    # conv7 = Dropout(0.2)(conv7)

    # Encoder Path
    inputs = Input(shape=(256, 256, 3))
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(pool3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv4)

    # Decoder Path with Dropout
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv3], axis=-1)
    conv5 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(up5)
    conv5 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv5)
    conv5 = Dropout(0.2)(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(up6)
    conv6 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv6)
    conv6 = Dropout(0.2)(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(up7)
    conv7 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(conv7)


    # Output
    outputs = Conv2D(21, (1, 1), activation='softmax')(conv7)

    # set optimiser
    optimiser = Adam(learning_rate=1e-2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimiser, loss=focal_tversky_loss, metrics=['accuracy', mean_iou])

    return model

def fit(model, X, Y, X_val, Y_val, callbacks):
    # Create the generator
    batch_size = 8
    train_gen = data_generator(X, Y, batch_size)

    # Train the model using the generator
    model.fit(train_gen, steps_per_epoch=len(X)//batch_size, epochs=30, validation_data=(X_val, Y_val), callbacks=callbacks)
    return model


if __name__ == "__main__":
    # read training data and train the model from prepped_data/train_images and prepped_data/train_segmentation
    train_X = np.array([np.array(Image.open(f"./working/train_X/{i}")) for i in os.listdir("./working/train_X/")])
    train_y_seg = np.array([np.array(Image.open(f"./working/train_y_seg/{i}")) for i in os.listdir("./working/train_y_seg/")])

    # load augmented data from working/train_X_aug and train_y_aug and concatenate with train_X and train_y_seg
    train_X_aug = np.array([np.array(Image.open(f"./working/train_X_aug/{i}")) for i in os.listdir("./working/train_X_aug/")])
    train_X = np.concatenate((train_X, train_X_aug), axis=0)

    train_y_seg_aug = np.array([np.array(Image.open(f"./working/train_y_seg_aug/{i}")) for i in os.listdir("./working/train_y_seg_aug/")])
    train_y_seg = np.concatenate((train_y_seg, train_y_seg_aug), axis=0)

    # load validation data from working/val_X and val_y_seg
    val_X = np.array([np.array(Image.open(f"./working/val_X/{i}")) for i in os.listdir("./working/val_X/")])
    val_y_seg = np.array([np.array(Image.open(f"./working/val_y_seg/{i}")) for i in os.listdir("./working/val_y_seg/")])

    # convert validation set to one-hot encoding
    val_y_seg = to_categorical(val_y_seg, num_classes=21)

    print(train_X.shape)
    print(train_y_seg.shape)

    model = build()

    # model.load_weights("./model-04.h5")


    # define a callback to save the model after every epoch
    filepath = "model-{epoch:02d}-cw.h5"
    checkpoint = ModelCheckpoint(filepath, save_weights_only=True, save_best_only=False)

    model = fit(model, train_X, train_y_seg, val_X, val_y_seg, callbacks=[checkpoint])

    # save the model
    model.save("./working/model.h5")



