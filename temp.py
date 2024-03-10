import os
import numpy as np
import keras
import tensorflow as tf
from keras.src.utils import load_img, img_to_array


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

test_images = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\val"
pb_model_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw"

new_model = tf.keras.models.load_model(pb_model_dir)
new_model.summary()



# Define the directory containing your test images
test_dir = test_images


# Function to load images from a directory
def load_images_from_dir(directory):
    images = []
    labels = []
    class_names = os.listdir(directory)
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = load_img(os.path.join(class_dir, filename), target_size=(256, 256))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_name)

    images = np.array(images)
    labels = np.array(labels)
    labels = label_encoder.transform(labels)

    return images, labels


# Load test images and labels
test_images, test_labels = load_images_from_dir(test_dir)

# Normalize pixel values to be between 0 and 1
test_images = test_images / 255.0

# Evaluate the model on test data
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

