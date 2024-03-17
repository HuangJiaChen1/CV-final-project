import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
# from keras.initializers.initializers import RandomNormal, Constant
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras import models, layers
import os

def load_or_preprocess():
    if os.path.exists('x_processed_train.npy') and os.path.exists('y_train.npy') and os.path.exists('kernels.npy'):
        print("Loading processed data...")
        x_processed_train = np.load('x_processed_train.npy')
        y_train = np.load('y_train.npy')
        all_kernels = np.load('kernels.npy')
    else:
        print("Processed data not found. Running preprocessing...")
        from preprocess import process_data
        x_processed_train, y_train, all_kernels = process_data()
        np.save('x_processed_train.npy', x_processed_train)
        np.save('y_train.npy', y_train)
        np.save('kernels.npy',all_kernels)

    return x_processed_train, y_train, all_kernels


x_train, y_train, all_kernels = load_or_preprocess()
print(y_train)
if __name__ == "__main__":
# Original Ver.
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(32, 32, 6)))
    # model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides= 2))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides= 2))
    # model.add(layers.Dropout(0.2))
    #
    # model.add(layers.Conv2D(512, (5,5), activation= 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(units= len(all_kernels),activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False), metrics= ['sparse_categorical_accuracy'])
    model.summary()

#ResNet Ver. (Layers frozen)
    # base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    # for layer in base_model.layers:
    #     layer.trainable = False
    # N = 3
    # for layer in base_model.layers[-N:]:
    #     layer.trainable = True
    # x = base_model.output
    # x = layers.GlobalAveragePooling2D()(x)
    # x = Dense(1024,kernel_initializer='he_normal', bias_initializer='zeros', activation='relu')(x)
    # predictions = Dense(len(all_kernels), activation='softmax',kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=Constant(value=0.1))(x)
    # model = models.Model(inputs=base_model.input, outputs=predictions)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #               metrics=['sparse_categorical_accuracy'])
    # model.summary()


    # cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoint/weights.ckpt", save_weights_only= True, save_best_only= True)
    # model.save_weights('w.weights.h5')

    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size= 64)
    model.save_weights('model')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['sparse_categorical_accuracy']
    val_accuracy = history.history['val_sparse_categorical_accuracy']
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