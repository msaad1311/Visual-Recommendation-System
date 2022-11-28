import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.layers import Flatten, AveragePooling2D
from tensorflow.keras import applications
from tensorflow.keras.models import Model


TYPE = "Apparel"
GENDER = "Boys"
PATH_IMAGE = f"../../data/raw/{TYPE}/{GENDER}/Images/"
PATH_FILE = r"../../data/raw/fashion.csv"
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
EPOCHS = 50
BATCH_SIZE = 1
RESNET_MODEL = applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
)
FILENAMES = []


output = RESNET_MODEL.output
output = AveragePooling2D(pool_size=(3, 3))(RESNET_MODEL.output)
output = AveragePooling2D(pool_size=(2, 2))(output)
output = Flatten()(output)
model = Model(inputs=RESNET_MODEL.input, outputs=output)


fashion_lookup = pd.read_csv(PATH_FILE)
sample_size = len(fashion_lookup[fashion_lookup["Gender"] == GENDER])


# Trying out the feature extraction
datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    # rescale= 1/255
)

generator = datagen.flow_from_directory(
    PATH_IMAGE,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)


for filename in generator.filenames:
    FILENAMES.append(filename.split("/")[-1])


image_features = model.predict_generator(
    generator, sample_size // BATCH_SIZE
)

# pairwise_dist = pairwise_distances(image_features)
# distance_dataframe = pd.DataFrame(
#     pairwise_dist, columns=FILENAMES, index=FILENAMES
# )


# for any image get the
FILENAMES.append("2694.jpg")
src_to_test_image = f"../../data/raw/{TYPE}/{GENDER}/Images/2694.jpg"

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


foobar = np.concatenate((image_features, prediction), axis=0)

pairwise_dist = pairwise_distances(foobar)

distance_dataframe = pd.DataFrame(
    pairwise_dist, columns=FILENAMES, index=FILENAMES
)

closest_imgs_scores = distance_dataframe["2694.jpg"].sort_values(
    ascending=True
)[1 : 5 + 1]
print(closest_imgs_scores)
