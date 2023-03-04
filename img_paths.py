

#data loader

import os
import sys
import random
from IPython.display import Image, display
import numpy as np
import random
from matplotlib import pyplot as plt

from trailClass import *

def get_img_path(input_dir, target_dir):

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            #if fname.endswith(".png") #and not fname.startswith(".")
            if fname.endswith("color.png") #and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))



    # randomly split the dataset into two
    # Split our img paths into a training and a validation set
    val_samples = 90
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)


    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)

    return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths