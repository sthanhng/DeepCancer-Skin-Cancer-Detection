import Augmentor

p = Augmentor.Pipeline("D:/Learnning/TensorFlow/AugmentationData/nhap/")

# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
#p.random_brightness(probability=1,min_factor=0.5, max_factor=1)
p.ground_truth("D:/Learnning/TensorFlow/AugmentationData/ground/")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)

p.resize(probability=1.0, width=128, height=128)
p.sample(20)