import os
import numpy as np
import tensorflow as tf
from keras.src.utils import load_img, img_to_array


# from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_validation_data_from_folder(folder_path, target_size=(224, 224)):
    X_val = []
    y_val = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Assuming the folder structure is such that each class has its own subfolder
        class_label = os.path.basename(folder_path)

        # Load image and preprocess
        image = load_img(file_path, target_size=target_size)
        image = img_to_array(image)
        image /= 255.0  # Normalize pixel values

        # Append image and label to the lists
        X_val.append(image)
        y_val.append(class_label)

    # Convert lists to NumPy arrays
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return X_val, y_val


# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\saved_model.pb")

# Parent directory containing folders of validation data
parent_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\val"

# Iterate through all folders in the parent directory
for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)

    # Check if the item is a directory
    if os.path.isdir(folder_path):
        # Load validation data from the current folder
        validation_data = load_validation_data_from_folder(folder_path)

        # Unpack validation data
        X_val, y_val = validation_data

        # Evaluate the model on validation data
        loss, accuracy = model.evaluate(X_val, y_val)

        # Print the results
        print(f'Validation Loss for {folder}: {loss:.4f}')
        print(f'Validation Accuracy for {folder}: {accuracy:.4f}')
