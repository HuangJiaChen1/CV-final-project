import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense


# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_built_with_cuda())
(x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

print(x_train[0].shape)

all_kernels = []
def generate_motion_blur_kernel(size, angle):
    kernel = np.zeros((30, 30))
    array = np.zeros((30,1), dtype=int)
    start_index = (30 - size) // 2
    array[start_index:start_index + size, :] = 1
    kernel[14,:] =array.flatten()
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((15,15), angle, 1.0), (30, 30))
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_motion_blur(img):
    global all_kernels
    all_blurred = []
    # Load the image
    img = np.clip(img, 0, 255)
    for kernel in all_kernels:
        blurred_img = cv2.filter2D(img, -1, kernel)
        all_blurred.append(blurred_img)
        '''
        Visualization
        '''
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(img)
        # plt.title('Original Image')
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(blurred_img)
        # plt.title('Motion Blurred Image')
        # plt.axis('off')
        #
        # plt.subplot(1,3,3)
        # plt.imshow(kernel)
        # plt.title('Kernel')
        # plt.axis('off')
        # plt.show()
    all_blurred = np.array(all_blurred)
    return all_blurred


#create all kernels
kernel = generate_motion_blur_kernel(size=1, angle=0)
all_kernels.append(kernel)
for angle in range(0,171,10):
    for size in range(2,31,2):
        kernel = generate_motion_blur_kernel(size=size, angle=angle)
        all_kernels.append(kernel)


y_train = np.zeros((x_train.shape[0],2))
print(y_train.shape)
y_test = np.zeros((x_test.shape[0],2))
for i in range(1):
    image = x_train[i]
    # print(image)
    blurred = apply_motion_blur(image)
    print(blurred.shape)
    # print(angle)


# for i in range(x_train.shape[0]):
#     image = x_train[i]
#     # print(image)
#     speed = int(np.random.uniform(0,8))
#     k = int(np.random.uniform(3, 20))
#     kernel,angle,blurred_img = apply_motion_blur(image, kernel_size= k)
#     # print(angle)
#     x_train[0] = blurred_img
#
#     ker = generate_motion_blur_kernel(k,angle)
#     # print(kernel-ker) # same
#     y_train[i][0] = k
#     y_train[i][1] = angle
#
# for i in range(x_test.shape[0]):
#     image = x_test[i]
#     # print(image)
#     speed = int(np.random.uniform(0,8))
#     k = int(np.random.uniform(3, 30))
#     kernel,angle,blurred_img = apply_motion_blur(image, kernel_size= k)
#     # print(angle)
#     x_test[0] = blurred_img
#
#     ker = generate_motion_blur_kernel(k,angle)
#     # print(kernel-ker) # same
#     y_test[i][0] = k
#     y_test[i][1] = angle
#
# if __name__ == "__main__":
#     model = models.Sequential()
#
#     # Convolutional layers
#     model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(32, 32, 3)))
#     model.add(BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.25))
#
#     model.add(layers.Conv2D(256, (5, 5), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.25))
#
#     model.add(layers.Conv2D(512, (5,5), activation= 'relu'))
#     model.add(layers.Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(layers.Dense(2, activation='linear'))
#
#     model.compile(optimizer='adam', loss='mse')
#     model.summary()
#     cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath= "./weights/w.weights.h5", save_weights_only= True, save_best_only= True)
#     model.save_weights('w.weights.h5')
#
#     history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks= [cp_callbacks])
#
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#
#     # Plotting the training and validation loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, loss, 'bo', label='Training loss')  # "bo" gives us blue dot
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')  # "b" is for "solid blue line"
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.show()