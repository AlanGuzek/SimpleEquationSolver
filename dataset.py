from PIL import Image
import numpy as np
import pandas as pd
import os


def get_dataset():
    dataset = []
    value = []
    path = 'resizedtrain'
    i = 0
    names = {}
    for folder in os.listdir(path):
        names[folder] = i
        for character in os.listdir(os.path.join(path, folder)):
            img = np.asarray(Image.open(os.path.join(path, folder, character)).convert('L'))
            img = img.flatten()
            dataset.append(img)
            value.append(i)
        i += 1
    df = pd.DataFrame(dataset)
    df['Values'] = value
    return df


get_dataset()
