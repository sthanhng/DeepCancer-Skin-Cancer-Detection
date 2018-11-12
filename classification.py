import numpy as np
import scipy.io as scio
import os,re
import itertools
import keras
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import *
from keras.layers import  Input, Flatten, Dense, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as keras

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_key(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

total_images = np.load('total_images.npy')
ground_truth_images = np.load('gt_labels_binary.npy')

root_path = "D:/Learnning/TensorFlow/models_example/skin_cancer_detection_segmentation/"

def get_filenames(path):
    filenames = []
    for root, dirnames, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        root_path = root
    print(len(filenames))
    return filenames

filenames_melanoma = get_filenames(root_path + "melanoma1_resized/")
filenames_others = get_filenames(root_path + "others1_resized/")

filenames_total = filenames_melanoma + filenames_others
filenames_total.sort(key=natural_key)

segmented_images = np.copy(total_images)
x, y, z = segmented_images[0].shape
print('x:{} y:{} z:{}'.format(x, y, z))
for i in range(len(total_images)):
    for j in range(x):
        for k in range(y):
            for l in range(z):
                print('i:{} j:{} k:{} l:{}'.format(i, j, k, l))
                segmented_images[i][j][k][l] = total_images[i][j][k][l] if ground_truth_images[i][j][k][l] == 1 else 0
    misc.imsave(root_path+"segmented_images/segmented_"+filenames_total[i], segmented_images[i])

segmented_images[0].shape

