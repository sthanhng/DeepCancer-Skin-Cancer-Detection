###########################################################
#
# DeepCancer - Skin cancer detection
# Description:
# - utils.py
# - create date: 2018-11-24
#
############################################################

import numpy as np
import scipy.io as scio
import os, re
import itertools
import keras
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import *
from keras.layers import Input, Flatten, Dense, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, \
    Concatenate, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as keras


# load dataset
train_images = np.load('classification_train_images.npy')
test_images = np.load('classification_test_images.npy')
train_labels = np.load('classification_train_labels.npy')
test_labels = np.load('classification_test_labels.npy')


def get_unet_model(image_dims):
    inputs = Input((image_dims[0], image_dims[1], image_dims[2]))
    conv1 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(128, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal') \
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    flatten1 = Flatten()(conv9)
    dense2 = Dense(1, activation='sigmoid')(flatten1)
    model = Model(inputs=inputs, outputs=dense2)

    ## return the model
    return model


def save_model_every_epoch(model):

    # print the summary of the model
    model.summary()

    # compile model
    model.compile(optimizer=Adam(lr=1 - 25),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    lr_reducer = ReduceLROnPlateau(factor=0.5,
                                   cooldown=0,
                                   patience=6,
                                   min_lr=0.5e-6)

    csv_logger = CSVLogger('Unet_classifier.csv')

    # create checkpoint
    checkpoint_fn = "models/Unet_{epoch:03d}_{val_acc:.3f}.hdf5"
    model_checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=1,
                                       save_best_only=False, period=1)

    model.fit(train_images, train_labels, batch_size=10, epochs=20, verbose=1,
              validation_data=(test_images, test_labels), shuffle=True,
              callbacks=[lr_reducer, csv_logger, model_checkpoint])


## Testing save_model_every_epoch()
if __name__ == '__main__':
    model = get_unet_model((128, 192, 3))
    save_model_every_epoch(model)
