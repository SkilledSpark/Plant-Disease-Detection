import os
import keras
import tensorflow as tf


pb_model_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw"
#
# # Loading the Tensorflow Saved Model (PB)
# model = tf.saved_model.load(pb_model_dir)

tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device=None,
    experimental_skip_checkpoint=False,
    experimental_variable_policy=None,
    experimental_load_function_aliases=False
)

model = tf.keras.saving.load_model(
    pb_model_dir, custom_objects=None, compile=True, safe_mode=True
)

model.save(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\final_model.keras")

