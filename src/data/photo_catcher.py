import numpy as np
import pandas as pd
import cv2
import os

TYPE = 'Apparel'
GENDER = 'Boys'
PATH_IMAGE = f'../../data/raw/{TYPE}/{GENDER}/Images/Images_with_product_ids/'
PATH_FILE = f'../../data/raw/fashion.csv'


temporary_image_name = os.listdir(PATH_IMAGE)[0]
print(temporary_image_name)
image = cv2.imread(PATH_IMAGE + temporary_image_name)
print(image.shape)
image_dictionary = {int(temporary_image_name.split('.')[0]): image}
image_table = pd.DataFrame(image_dictionary)

fashion_lookup = pd.read_csv(PATH_FILE)
fashion_lookup = fashion_lookup[fashion_lookup['ProductId']== int(temporary_image_name.split('.')[0])]
final_fashion_table = pd.merge()



