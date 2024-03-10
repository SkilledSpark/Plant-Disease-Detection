import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pathlib

class CustomLayer(keras.layers.Layer):
    def __init__(self, arg1, arg2, **kwargs):
        super().__init__(**kwargs)
        self.arg1 = arg1
        self.arg2 = arg2

    def get_config(self):
        # Call the base class method to get the base config
        config = super().get_config()
        # Update the config with our custom arguments
        config.update({
            "arg1": self.arg1,
            "arg2": self.arg2
        })
        return config

# Training dir
data_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\train"
data_dir = pathlib.Path(data_dir)
print(data_dir)

batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir)
class_names=train_ds.class_names


# Testing dir
test_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\val"
test_dir = pathlib.Path(test_dir)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir)

# One hot encoding
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=38)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=38)))


# # Plotting
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(6):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#
# plt.show()


# Training
resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(256,256,3),
                   pooling='avg',classes=38,
                   weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(38, activation='softmax'))

resnet_model.summary()
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

epochs=1
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
resnet_model.save(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\model.h5")
resnet_model.save(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\modelextra.keras")
