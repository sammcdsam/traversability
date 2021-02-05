import os
import sys
import cv2
import models.led

input_dir = "datasets/traversability/May_set/img_May/image_drive"
target_dir = "datasets/traversability/May_set/img_May/annotations_drive"
path = 'outputs/gifs/LEDNet_1.gif'
img_size = (480, 640)
num_classes = 4
batch_size = 8

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
        if fname.endswith(".png") #and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)


from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

# Display input image #7
im = load_img(input_img_paths[9])
#im.show()

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
#img.show()


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
np.set_printoptions(threshold=sys.maxsize)

class trailClass(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            
        
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            
            y[j] = np.expand_dims(img, 2)

            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        return x, y


from tensorflow.keras import layers


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
#model = get_model(img_size, num_classes)
#model.summary()

#load model
model = keras.models.load_model('trained_models/model.h5')

import random
from matplotlib import pyplot as plt
import cv2 as cv
from matplotlib import cm
import webcolors
# Split our img paths into a training and a validation set

val_samples = 75
val_input_img_paths = input_img_paths
val_target_img_paths = target_img_paths


val_gen = trailClass(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    img.show()


# Display results for validation image #10
i = 0
img_array = []
color_count = {}

for i in range(val_samples):
    orig = load_img(val_input_img_paths[i])
    label = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i+1]))

    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    

    

    
    color_map = np.array([
        [145, 208, 80],
        [56,86,34],
        [236,124,48],
        [134,206,234],
        [255,192,0]
    ])

    #print(list(keras.preprocessing.image.array_to_img(mask).getdata()))
    mask_img = PIL.ImageOps.colorize(keras.preprocessing.image.array_to_img(mask), [134, 206, 235],  [255, 192, 0], [56, 86, 34], midpoint = 62)

    #mask_img = mask_img.convert("RGB")

    width, height = orig.size
    mask_img = mask_img.resize((width,height))


    mask = PIL.Image.new("L", orig.size, 128)

    final = PIL.Image.composite(orig, mask_img, mask)
    
    img_array.append(final)

print(color_count)
im.save(path, save_all=True, append_images=img_array)


    #titles = ['Original Image','Labeled Image','Output']
    #images = [orig, label, mask_img]
    #for j in range(3):
    #    plt.subplot(1,3,j+1),plt.imshow(images[j])
    #    plt.title(titles[j])
    #    plt.xticks([]),plt.yticks([])
    #plt.show()
