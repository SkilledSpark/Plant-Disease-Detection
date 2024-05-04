import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pathlib

# Training dir
data_dir = r'C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\train'
data_dir = pathlib.Path(data_dir)

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
class_names = train_ds.class_names

# Testing dir
test_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\val"
test_dir = pathlib.Path(test_dir)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir)

# One hot encoding
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=38)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=38)))

# Training
inputs = tf.keras.Input(shape=(256, 256, 3))
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_tensor=inputs,
                                                  pooling='avg',
                                                  weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

x = pretrained_model.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(38, activation='softmax')(x)

resnet_model = Model(inputs=inputs, outputs=outputs)
resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

epochs = 1
history = resnet_model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=epochs)

resnet_model.save(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\model.h5")
resnet_model.save(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\modelextra.keras")
