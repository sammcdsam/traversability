import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from keras_segmentation.models.segnet import segnet
from keras_segmentation.predict import predict



# city scape set - 33 classes
#input_dir = "datasets/cityscape/train_images/train"
#target_dir = "datasets/cityscape/train_segmentation/train"
#val_dir = "datasets/cityscape/val_images/val"
#val_target_dir = "datasets/cityscape/val_segmentation/val"

# traversability set - 4 classes
#input_dir = "datasets/traversability/all_sets/train_images"
#target_dir = "datasets/traversability/all_sets/train_masks"
#val_dir = "datasets/traversability/all_sets/val_images"
#val_target_dir = "datasets/traversability/all_sets/val_masks"

# MAVS set - 5 classes - 480x640
#input_dir = "datasets/MAVS/all_img"
#target_dir = "datasets/MAVS/all_annotated"
#val_dir = "datasets/MAVS/val_img"
#val_target_dir = "datasets/MAVS/val_annotated"

input_dir = "datasets/Rellis-3D/images"
target_dir = "datasets/Rellis-3D/annotations"
val_dir = "datasets/Rellis-3D/val_img"
val_target_dir = "datasets/Rellis-3D/val_an"

model = segnet(n_classes=18,  input_height=600, input_width=960 )

#history = model.train( 
    #verify_dataset = False,
    #train_images =  input_dir,
    #train_annotations = target_dir,
    #validate=True,
    #val_images = val_dir,
    #val_annotations = val_target_dir,
    #do_augment = False,
    #checkpoints_path = "checkpoints/checkpoints_RELLIS-3D_dataset/rellis-3d" , epochs=150, batch_size=4, steps_per_epoch=512)

model.load_weights("checkpoints/checkpoints_RELLIS-3D_dataset/rellis-3d.00150")

out = model.predict_segmentation(
    #inp="datasets/traversability/all_sets/val_images/1562093067.865790592.png",
    #inp="datasets/cityscape/val_images/val/frankfurt_000000_003357.png",
    #inp="datasets/Rellis-3D/images/00004/frame001019-1581791780_308.jpg",
    inp="1562093031.865889600.jpg",
    out_fname="output.png"
)


'''
# summarize history for loss
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
plt.close()


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")
plt.close()


# summarize history for loss
plt.figure()
plt.plot(history.history['mean_io_u'])
plt.plot(history.history['val_mean_io_u'])
plt.title('model mean iou')
plt.ylabel('mean iou')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("mean_io_u.png")
plt.close()
'''
