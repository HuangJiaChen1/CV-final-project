import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import os
from tensorflow.keras.optimizers import Adam

def load_or_preprocess():
    if os.path.exists('x_processed_train.npy') and os.path.exists('y_train.npy'):
        print("Loading processed data...")
        x_processed_train = np.load('x_processed_train.npy')
        y_train = np.load('y_train.npy')
    else:
        print("Processed data not found. Running preprocessing...")
        from preprocess import process_data
        x_processed_train, y_train = process_data()
        np.save('x_processed_train.npy', x_processed_train)
        np.save('y_train.npy', y_train)

    return x_processed_train, y_train


x_train, y_train = load_or_preprocess()
print(y_train)
if __name__ == "__main__":
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (5,5), activation= 'relu'))
    model.add(layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(layers.Dense(271, activation='softmax'))

    optimizer = Adam(learning_rate= 0.04)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= ['accuracy'])
    model.summary()
    cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath= "./weights/w.weights.h5", save_weights_only= True, save_best_only= True)
    # model.save_weights('w.weights.h5')

    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size= 64, callbacks= [cp_callbacks])

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'bo', label='Training loss')  # "bo" gives us blue dot
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  # "b" is for "solid blue line"
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')  # "ro" gives us red dot
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')  # "r" is for "solid red line"
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()