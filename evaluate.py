# import tensorflow as tf

# model = tf.keras.models.load_model("models/emotion_mobilenet.h5")

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# test_datagen = ImageDataGenerator(rescale=1./255)

# test_gen = test_datagen.flow_from_directory(
#     "data/test",
#     target_size=(96,96),
#     color_mode="rgb",
#     class_mode="categorical",
#     batch_size=64
# )

# loss, acc = model.evaluate(test_gen)
# print("Test Accuracy:", acc)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model = tf.keras.models.load_model("models/emotion_mobilenet.h5")

model = tf.keras.models.load_model("models/emotion_mobilenet_final.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    "data/test",
    target_size=(96,96),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

steps = test_gen.samples // test_gen.batch_size
loss, acc = model.evaluate(test_gen, steps=steps)

print("Test Accuracy:", round(acc * 100, 2), "%")
