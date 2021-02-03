from PIL import Image
import numpy as np
import pandas as pd
import os


def get_dataset():
    dataset = []
    path = 'resizedtrain'
    i = 0
    names: dict = {}
    for folder in os.listdir(path):
        names[folder] = i
        i += 1
        for character in os.listdir(os.path.join(path, folder)):
            img = np.asarray(Image.open(os.path.join(path, folder, character)).convert('L'))
            img = img.flatten()
            temp: list = []
            for b in img:
                temp.append(b)
            temp.append(i)
            dataset.append(temp)
    df = pd.DataFrame(dataset)
    df.to_pickle("data_frame.pkl")
    return df


collected_data = get_dataset()
