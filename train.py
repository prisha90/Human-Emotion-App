import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

img_size = 96
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "data/train",
    target_size=(img_size,img_size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size
)

test_gen = test_datagen.flow_from_directory(
    "data/test",
    target_size=(img_size,img_size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size
)

base_model = MobileNetV2(
    input_shape=(img_size,img_size,3),
    include_top=False,
    weights="imagenet"
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20
)

model.save("models/emotion_mobilenet.h5")
