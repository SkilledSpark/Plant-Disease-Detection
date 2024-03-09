import os
import shutil
from sklearn.model_selection import train_test_split

# Define the parent directory containing the folders for each class
parent_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented"

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate through each folder in the parent directory
for class_name in os.listdir(parent_dir):
    class_dir = os.path.join(parent_dir, class_name)
    if os.path.isdir(class_dir):
        # Iterate through the image files in each class directory
        for filename in os.listdir(class_dir):
            # Construct the full file path
            file_path = os.path.join(class_dir, filename)
            # Add the file path and label to the lists
            file_paths.append(file_path)
            labels.append(class_name)

# Split the dataset into training and validation sets (80-20 split)
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42)

# Create directories for training and validation sets
train_dir = os.path.join(parent_dir, 'train')
val_dir = os.path.join(parent_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Move training files to the train directory
for file_path, label in zip(train_files, train_labels):
    destination = os.path.join(train_dir, label)
    os.makedirs(destination, exist_ok=True)
    shutil.move(file_path, destination)

# Move validation files to the val directory
for file_path, label in zip(val_files, val_labels):
    destination = os.path.join(val_dir, label)
    os.makedirs(destination, exist_ok=True)
    shutil.move(file_path, destination)

print("Files moved successfully.")
