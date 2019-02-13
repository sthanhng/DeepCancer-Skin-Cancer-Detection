###########################################################
#
# DeepCancer - Skin cancer detection
# Description:
# - segmentation.py
# - create date: 2019-2-1
#
############################################################


import numpy as np
import argparse
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.optimizers import *
from Unet_Model import UNetModel
from utils import get_filenames, natural_key, atoi
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2

class Segmentation():
    def __init__(self, batch_size, epochs, data_path, image_train_dir, mask_train_dir, image_eval_dir, mask_eval_dir,
                 weight_save_prefix='Unet_weight_segmentation.hdf5', save_csv_logger='Unet_logger_segmentation.csv',
                 save_to_dir=None, seed=1):
        self.batch_size = batch_size
        self.data_path = data_path
        self.epochs = epochs
        self.image_train_dir = image_train_dir
        self.mask_train_dir = mask_train_dir
        self.image_eval_dir = image_eval_dir
        self.mask_eval_dir = mask_eval_dir
        self.weight_save_prefix = weight_save_prefix
        self.save_csv_logger = save_csv_logger
        self.save_to_dir = save_to_dir
        self.seed = seed

    def _Model_Unet_segmentation(self):
        # training segmentation the model Unet
        model = UNetModel.get_unet_model_seg((128, 128, 3))
        model.summary()
        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        lr_reducer = ReduceLROnPlateau(factor=0.5,
                                       cooldown=0,
                                       patience=6,
                                       min_lr=0.5e-6)

        csv_logger = CSVLogger(save_csv_logger)

        model_checkpoint = ModelCheckpoint(weight_save,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        model.fit(image_train,
                  mask_train,
                  batch_size=4,
                  epochs=30,
                  verbose=1,
                  validation_data=(image_eval, mask_eval),
                  shuffle=True,
                  callbacks=[lr_reducer, csv_logger, model_checkpoint])

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='D:/Learnning/TensorFlow/program/model_base_Resnet/DATA/_data_input/_data_segmentation/', type=str,
                        help='The path to the training data')
    parser.add_argument('--image-train-dir', default='_data_train/_image_input/', type=str,
                        help='The path to the images train')
    parser.add_argument('--mask-train-dir', default='_data_train/_groundtruth/', type=str,
                        help='The path to the masks train')
    parser.add_argument('--image-eval-dir', default='_data_val/_image_eval/', type=str,
                        help='The path to the images val')
    parser.add_argument('--mask-eval-dir', default='_data_val/_groundtruth_eval/', type=str,
                        help='The path to the masks val')
    parser.add_argument('--weight-save-prefix', default='_model-weights/segmentation/Unet_weight_segmentation.hdf5', type=str,
                        help='The name to save weight prefix')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='The batch size of the ')
    parser.add_argument('--epochs', default=30, type=int,
                        help='The epochs of the ')
    parser.add_argument('--save-csv-logger', default='_output_data/segmentation/Unet_logger_segmentation.csv', type=str,
                        help='The name to save csv logger')
    parser.add_argument('--save-to-dir', default='D:/Learnning/TensorFlow/program/model_base_Resnet/OUTPUT/', type=str,
                        help='The path to the save')
    return parser.parse_args()


if __name__ == "__main__":
    # ================= get the arguments ====================
    args = _get_args()
    save_csv_logger = os.path.join(args.save_to_dir, args.save_csv_logger)
    weight_save = os.path.join(args.save_to_dir, args.weight_save_prefix)

    image_train_dir = os.path.join(args.data_path, args.image_train_dir)
    mask_train_dir = os.path.join(args.data_path, args.mask_train_dir)
    image_eval_dir = os.path.join(args.data_path, args.image_eval_dir)
    mask_eval_dir = os.path.join(args.data_path, args.mask_eval_dir)

    # # ========================================================
    # # create image train
    filenames_train = get_filenames(image_train_dir)
    filenames_train.sort(key=natural_key)
    image_train = []
    for file in filenames_train:
        image_train.append(ndimage.imread(image_train_dir + file))
    image_train = np.array(image_train)
    print(image_train[0].shape)

    # # ========================================================
    # # create mask train
    filenames_train = get_filenames(mask_train_dir)
    filenames_train.sort(key=natural_key)
    mask_train = []
    for file in filenames_train:
        mask_train.append(ndimage.imread(mask_train_dir + file))

    np.unique(mask_train[0])
    gt_labels_binary = []
    for gt_image in mask_train:
        ret, image = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
        gt_labels_binary.append(image)
    gt_labels_binary = np.array(gt_labels_binary)
    np.unique(gt_labels_binary[0])
    gt_labels_binary = gt_labels_binary / 255
    np.unique(gt_labels_binary[0])
    mask_train = gt_labels_binary


    # # ========================================================
    # # create image eval
    filenames_eval = get_filenames(image_eval_dir)
    filenames_eval.sort(key=natural_key)
    image_eval = []
    for file in filenames_eval:
        image_eval.append(ndimage.imread(image_eval_dir + file))
    image_eval = np.array(image_eval)

    # # ========================================================
    # # create mask eval
    filenames_eval = get_filenames(mask_eval_dir)
    filenames_eval.sort(key=natural_key)
    mask_eval = []
    for file in filenames_eval:
        mask_eval.append(ndimage.imread(mask_eval_dir + file))
    mask_eval = np.array(mask_eval)

    np.unique(mask_eval[0])
    gt_labels_binary = []
    for gt_image in mask_eval:
        ret, image = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
        gt_labels_binary.append(image)
    gt_labels_binary = np.array(gt_labels_binary)
    np.unique(gt_labels_binary[0])
    gt_labels_binary = gt_labels_binary / 255
    np.unique(gt_labels_binary[0])
    mask_eval = gt_labels_binary

    # # ========================================================
    # # preproccesing

    train_mean = np.mean(image_train, axis=(0, 1, 2, 3))
    train_std = np.std(image_train, axis=(0, 1, 2, 3))
    image_train = (image_train - train_mean) / (train_std + 1e-7)

    mask_train = np.expand_dims(mask_train, axis=3)

    evaluate_mean = np.mean(image_eval, axis=(0, 1, 2, 3))
    evaluate_std = np.std(image_eval, axis=(0, 1, 2, 3))
    image_eval = (image_eval - evaluate_mean) / (evaluate_std + 1e-7)

    mask_eval = np.expand_dims(mask_eval, axis=3)
    # ========================================================
    # create the instance of Segmentation class
    aug = Segmentation(args.batch_size,
                       args.epochs,
                       args.data_path,
                       args.image_train_dir,
                       args.mask_train_dir,
                       args.image_eval_dir,
                       args.mask_eval_dir,
                       args.weight_save_prefix,
                       args.save_csv_logger,
                       args.save_to_dir)

    trainning = aug._Model_Unet_segmentation()
print('')
print('#==============================================#')
print('Congratulations on your successful training')
print('#==============================================#')