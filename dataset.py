from PIL import Image
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

dataset = []
path = 'eval'
for folder in os.listdir(path):
    for character in os.listdir(os.path.join(path, folder)):
        im = np.asarray(Image.open(os.path.join(path, folder, character)).convert('L'))
        im.flatten()
        im = np.append(im, character)
        dataset.append(im)
# data = pd.DataFrame(dataset)
pass
