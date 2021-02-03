from PIL import Image
import os

SIZE: tuple = (50, 50)
PIN: tuple = (0, 0)
COLOR: tuple = (255, 255, 255, 0)


def resize_image(img_path: str):
    image: Image = Image.open(img_path)
    image.thumbnail(SIZE, Image.ANTIALIAS)
    bg: Image = Image.new('RGBA', SIZE, COLOR).convert('L')
    bg.paste(image, PIN)
    return bg

# CODE USED IN PREPARING DATA FOR MACHINE TO LEARN
# path = 'train'
# path2 = 'resizedtrain'
# for folder in os.listdir(path):
#     for character in os.listdir(os.path.join(path, folder)):
#         img = Image.open(os.path.join(path, folder, character))
#         img.thumbnail(size, Image.ANTIALIAS)
#         bg = Image.new('RGBA', size, (255, 255, 255, 0)).convert('L')
#         bg.paste(img, (0, 0))
#         bg.save(os.path.join(path2, folder, character))
