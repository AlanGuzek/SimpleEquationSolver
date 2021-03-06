#  Changes image to DataFrame which can be used in model

# Importing
from pandas import DataFrame
from pathlib import Path
from img_resize import resize_image
import numpy as np
import pickle
from sign_dictionary import sign_dictionary

image_path: Path = Path("hand_draw/image.jpg")


# Function
def get_binary_image(path):
    img_list: list = []
    img = np.asarray(resize_image(path))
    img = img.flatten()
    img_list.append(img)
    img_df: DataFrame = DataFrame(img_list)
    return img_df


def get_model():
    return pickle.load(open('model2.pkl', 'rb'))


def check_image(model, b_image):
    ans = model.predict(b_image)
    print(sign_dictionary[int(ans)])


if __name__ == "__main__":
    check_image(get_model(), get_binary_image(image_path))
