import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import training
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (5, 5), activation='relu'))
    model.add(layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(layers.Dense(271, activation='softmax'))
    return model

model = create_model()
model.load_weights("./weights/w.weights.h5")
x_test = training.x_train[:10]
predictions = model.predict(x_test)
for i, prediction in enumerate(predictions):
    print(f"Predicted parameters for sample {i}: {prediction}")
    # image = x_test[i]
    # k = int(prediction[0])
    # angle = prediction[1]
    # kernel = training.generate_motion_blur_kernel(k,angle)
    # blurred_img = cv2.filter2D(image, -1, kernel)
    print(tf.argmax(prediction))
    print(training.y_train[i])
