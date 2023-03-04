import cv2
import numpy as np

from keras.preprocessing.image import img_to_array


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_path, mask_path, batch_size, dims, n_labels):
    while True:
        imgs = []
        labels = []
        for input_path, target_path in zip(img_path[:batch_size], mask_path[:batch_size]):
            # images
            original_img = cv2.imread(input_path)[:, :, ::-1]
            resized_img = cv2.resize(original_img, dims)
            array_img = img_to_array(resized_img) / 255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(target_path)
            resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            array_mask = category_label(resized_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels
