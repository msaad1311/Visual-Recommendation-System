"""
Script to run the similarity
"""
import pandas as pd
import numpy as np

from itertools import product
from sklearn.metrics import pairwise_distances
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.layers import Flatten, AveragePooling2D
from tensorflow.keras import applications
from tensorflow.keras.models import Model


TYPE = ["Footwear", "Apparel"]
GENDER = ["Boys", "Girls", "Men", "Women"]
PATH_FILE = r"../../data/raw/fashion.csv"
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
EPOCHS = 50
BATCH_SIZE = 1
RESNET_MODEL = applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
)
FILENAMES, IMAGE_FEATURES = [], []

output = RESNET_MODEL.output
output = AveragePooling2D(pool_size=(3, 3))(RESNET_MODEL.output)
output = AveragePooling2D(pool_size=(2, 2))(output)
output = Flatten()(output)
model = Model(inputs=RESNET_MODEL.input, outputs=output)

fashion_lookup = pd.read_csv(PATH_FILE)

# Trying out the feature extraction
datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    # rescale= 1/255
)

for types, gender in product(TYPE, GENDER):
    print(types, gender)
    PATH_IMAGE = f"../../data/raw/{types}/{gender}/Images/"
    sample_size = len(
        fashion_lookup[fashion_lookup["Gender"] == gender]
    )
    try:
        generator = datagen.flow_from_directory(
            PATH_IMAGE,
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=False,
        )
        for filename in generator.filenames:
            FILENAMES.append(PATH_IMAGE + filename)

        IMAGE_FEATURES.append(
            model.predict_generator(
                generator, sample_size // BATCH_SIZE
            )
        )
    except FileNotFoundError as e:
        continue

image_features_array = np.concatenate((IMAGE_FEATURES), axis=0)

# for any image get the
FILENAMES = FILENAMES[:-1]
FILENAMES.append("2691.jpg")
src_to_test_image = f"../../data/raw/Apparel/Boys/Images/2691.jpg"

# load the image
test_image = load_img(
    src_to_test_image, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
)

# preprocess the image
test_image = img_to_array(test_image)
test_image = test_image.reshape(
    (1, test_image.shape[0], test_image.shape[1], test_image.shape[2])
)

# make the prediction
prediction = model.predict(test_image)


foobar = np.concatenate((image_features_array, prediction), axis=0)

pairwise_dist = pairwise_distances(foobar)

distance_dataframe = pd.DataFrame(
    pairwise_dist, columns=FILENAMES, index=FILENAMES
)

closest_imgs_scores = distance_dataframe["2691.jpg"].sort_values(
    ascending=True
)[1 : 5 + 1]
print(closest_imgs_scores)
