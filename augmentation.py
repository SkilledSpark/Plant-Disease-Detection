import os
import numpy as np
import scipy
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Define the directory containing the image folders
base_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented"

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Iterate through the image folders
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        # Count the number of images in the folder
        num_images = len(os.listdir(folder_path))

        # Perform data augmentation if the number of images is less than 5000
        if num_images > 5000:
            print(f"Augmenting images in folder: {folder_name}")
            image_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
                           fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')]

            # Create directory to store augmented images
            augmented_dir = os.path.join(folder_path, 'augmented')
            os.makedirs(augmented_dir, exist_ok=True)

            # Load each image, perform augmentation, and save augmented images
            for img_file in image_files:
                img = load_img(img_file)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug',
                                          save_format='jpg'):
                    i += 1
                    if i >= 2:  # Generate 10 augmented images per original image
                        break  # Break the loop to avoid generating too many images per original image
