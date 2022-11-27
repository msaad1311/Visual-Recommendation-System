import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
output = RESNET_MODEL.output
output = AveragePooling2D(pool_size=(7, 7))(RESNET_MODEL.output)
output = Flatten()(output)
model = Model(inputs=RESNET_MODEL.input, outputs=output)

FILENAMES = []

fashion_lookup = pd.read_csv(PATH_FILE)
sample_size = len(fashion_lookup[fashion_lookup["Gender"] == GENDER])

# Trying out the feature extraction
datagen = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
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
