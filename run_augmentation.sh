#!/usr/bin/env bash

python augmentation.py \
    --data-path 'datasets/train/tr' \
    --image-dir 'images' \
    --mask-dir 'masks' \
    --save-dir 'augmented' \
    --batch-size 5 \
    --rotation-range 0.2 \
    --width-shift-range 0.05 \
    --height-shift-range 0.05 \
    --shear-range 0.05 \
    --zoom-range 0.05
