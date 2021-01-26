import os
import sys

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from tensorflow.keras.preprocessing.image import load_img

import random
from matplotlib import pyplot as plt

from trailClass import *
import unet_model


input_dir = "datasets/traversability/all_sets/images"
target_dir = "datasets/traversability/all_sets/annotations"
img_size = (608, 960)
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


# Display input image #7
im = load_img(input_img_paths[9])
#im.show()

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
#img.show()


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = unet_model.unet_model(img_size, num_classes)
model.summary()



# Split our img paths into a training and a validation set
val_samples = 90
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]


# Instantiate data Sequences for each split
train_gen = trailClass(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = trailClass(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['acc'])

callbacks = [
    keras.callbacks.ModelCheckpoint("der.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)


# Generate predictions for all images in the validation set

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

for i in range(15):
    orig = load_img(val_input_img_paths[i])
    label = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i+1]))

    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask_img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))


    titles = ['Original Image','Labeled Image','Output']
    images = [orig, label, mask_img]
    for j in range(3):
        plt.subplot(1,3,j+1),plt.imshow(images[j])
        plt.title(titles[j])
        plt.xticks([]),plt.yticks([])
    plt.show()

# summarize history for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
