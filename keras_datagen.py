import cv2
import numpy as np

def data_gen(image_list, mask_list, batch_size):
  c = 0
  n = image_list #List of training images
  
  while (True):
    img = np.zeros((batch_size, 512, 512, 3)).astype('float')
    mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      train_img = cv2.imread(n[i])/255.
      train_img =  cv2.resize(train_img, (512, 512))# Read an image from folder and resize
      
      img[i-c] = train_img #add to array - img[0], img[1], and so on.
                                                   

      train_mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE)/255.
      train_mask = cv2.resize(train_mask, (512, 512))
      train_mask = train_mask.reshape(512, 512, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

      mask[i-c] = train_mask

    c+=batch_size
    if(c+batch_size>=len((img_folder))):
      c=0
    yield img, mask




#train_frame_path = '/path/to/training_frames'
#train_mask_path = '/path/to/training_masks'

#val_frame_path = '/path/to/validation_frames'
#val_mask_path = '/path/to/validation_frames'

# Train the model
#train_gen = data_gen(train_frame_path,train_mask_path, batch_size = 4)
#val_gen = data_gen(val_frame_path,val_mask_path, batch_size = 4)