import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import training
all_kernels = training.all_kernels
def create_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 6)))
    # model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    #
    # model.add(layers.Conv2D(512, (5,5), activation= 'relu'))
    model.add(layers.Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(layers.Dense(len(all_kernels), activation='softmax'))
    #
    # base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    # for layer in base_model.layers:
    #     layer.trainable = False
    # N = 3
    # for layer in base_model.layers[-N:]:
    #     layer.trainable = True
    # x = base_model.output
    # x = layers.GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # predictions = Dense(len(all_kernels), activation='softmax')(x)
    # model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model()
model.load_weights("./checkpoint/weights.ckpt")
x_test = np.concatenate((training.x_train[0:100],training.x_train[272:373]),axis= 0)
print(x_test[0])
predictions = model.predict(x_test)
n = 0
correct = 0
for i, prediction in enumerate(predictions):
    # print(f"Predicted parameters for sample {i}: {prediction}")
    # image = x_test[i]
    # k = int(prediction[0])
    # angle = prediction[1]
    # kernel = training.generate_motion_blur_kernel(k,angle)
    # blurred_img = cv2.filter2D(image, -1, kernel)
    out_class = np.argmax(prediction)
    # print(out_class)
    # print(prediction[out_class])
    # print(training.y_train[i])
    # print(prediction[training.y_train[i][0]])
    n += 1
    if prediction[out_class] == prediction[training.y_train[i][0]]:
        correct +=1
    # Visualization
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(all_kernels[out_class])
    # plt.title('Predicted Kernel')
    # plt.axis('off')
    #
    # plt.subplot(1,3,2)
    # plt.imshow(all_kernels[training.y_train[i][0]])
    # plt.title("Ground Truth Kernel")
    # plt.axis('off')
    #
    # plt.subplot(1,3,3)
    # plt.imshow(x_test[i][:,:,:3])
    # plt.title("train image")
    # plt.axis('off')
    # plt.show()
print('accuracy: ', correct/n)