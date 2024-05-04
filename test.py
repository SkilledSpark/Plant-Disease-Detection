from keras.preprocessing import image
import numpy as np
from keras.models import load_model

img_path = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\val\Tomato___Bacterial_spot\0b37769a-a451-4507-a236-f46348e3a9ac___GCREC_Bact.Sp 3265_final_masked.jpg"
img = image.load_img(img_path)

model = load_model(r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented\final\model.h5")

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
predictions = model.predict(x)
predicted_class_index = np.argmax(predictions)
print(predictions)
print(predicted_class_index)

class_labels = {
    0: 'Early Blight Potato',
    1: 'Healthy Potato',
    2: 'Late Blight Potato',
    3: 'Bacterial Spot Tomato',
    4: 'Early Blight Tomato',
    5: 'Healthy Tomato',
    6: 'Late Blight Tomato'
}

if predicted_class_index in class_labels:
    print(class_labels[predicted_class_index])
else:
    print("Class index not found in dictionary")
