from PIL import Image
import numpy as np
import pandas as pd
import os
import csv


dataset = []
value = []
path = 'resizedtrain'
for folder in os.listdir(path):
    for character in os.listdir(os.path.join(path, folder)):
        img = np.asarray(Image.open(os.path.join(path, folder, character)).convert('L'))
        img = img.flatten()
        dataset.append(img)
        value.append(folder)
dataset = np.asarray(dataset, dtype='object')
df = pd.DataFrame({"Data": dataset, "Value": value})
df.to_csv('dataset.csv')

pass
