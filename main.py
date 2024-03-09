import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


train_dir = r"C:\Users\Lakshya Singh\Documents\GitHub\PlantVillage-Dataset\raw\segmented"
test_dir="data_distribution_for_SVM/test"
diseases = os.listdir(train_dir)
print(diseases)
print(len(diseases))

nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_dir + '/' + disease))

img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
print(img_per_class)

index = [n for n in range(38)]
plt.figure(figsize=(20, 5))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')
plt.show()
