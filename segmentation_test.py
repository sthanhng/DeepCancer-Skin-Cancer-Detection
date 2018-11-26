from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
# load image
test_images = np.load(root_path+'OUTPUT/segmentation_test_image.npy')
print('test_images.shape', test_images.shape)

filenames_test = get_filenames(root_path+"DATA/data_test/")
filenames_test.sort(key=natural_key)
ix = 0
# load weights model
model = UNetModel.get_unet_model_seg((128, 128, 3))
model.load_weights(root_path+'OUTPUT/Unet_lr_e4_bs_4.hdf5')

sample_predictions = model.predict(test_images[ix].reshape((1, 128, 128, 3)))
sample_predictions = sample_predictions.reshape((128, 128))
# sample_predictions = sample_predictions > 0.5
# sample_predictions = np.array(sample_predictions, dtype=np.uint8)


cv2.imwrite(root_path+"DATA/segmented_images_train/segmented_"+filenames_test[ix], sample_predictions, (128, 128, 1))
plt.figure()
plt.imshow(sample_predictions, cmap="gray")
plt.figure()
abc = cv2.imread(root_path+"DATA/segmented_images_train/segmented_"+filenames_test[ix])
plt.imshow(abc, cmap="gray")
plt.show()
