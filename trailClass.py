#
# Create numpy arrays out of the image data. 
#
#

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random

#print out all numpy values
#np.set_printoptions(threshold=sys.maxsize)


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
        
        img_aug_array = []

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            
            # randomly flip some of the images.
            # will be removed when a better way to augment images is figured out. 
            if random.random()<20:
                flip = True
            else:
                flip = False
            
            if flip:
                np.flip(x[j], 1) 
                img_aug_array.append(flip)
            
        
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")

            y[j] = np.expand_dims(img, 2)
            if img_aug_array[j]:
                np.flip(y[j], 1)
            #print(np.unique(y[j]))
            uni = np.unique(y[j])
            for l in range(len(uni)):
                #create unique identifiers for each color in the label image/
                y[j][y[j]==uni[l]] = l


            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        return x, y