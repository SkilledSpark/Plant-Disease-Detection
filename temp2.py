import tensorflowjs as tfjs

# Path to the TensorFlow SavedModel or Keras HDF5 model
model_path = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\model.h5"

# Output directory where the converted TensorFlow.js model will be saved
output_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final"

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model_path, output_dir)
