import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense


# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_built_with_cuda())
(x_train, t), (x_test, _) = tf.keras.datasets.cifar100.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
print("Training data shape:", x_train.shape)
print("Testing data shape:", t.shape)

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

def process_data():
    #create all kernels
    kernel = generate_motion_blur_kernel(size=1, angle=0)
    all_kernels.append(kernel)
    for angle in range(0,171,10):
        for size in range(2,31,2):
            kernel = generate_motion_blur_kernel(size=size, angle=angle)
            all_kernels.append(kernel)

    x_processed_train = []
    orig_img = []
    # define how many images to be selected here
    num_indices = 100
    y_train = np.zeros((num_indices*271,1),dtype= int)
    sequence = np.arange(271)
    y_train = np.tile(sequence, num_indices).reshape((num_indices*271, 1))
    random_indices = np.random.choice(x_train.shape[0], num_indices, replace=False)

    print("y train: ", y_train.shape)
    y_test = np.zeros((x_test.shape[0],2))
    num = 0
    for i in random_indices:
        image = x_train[i]
        # for j in range(len(all_kernels)):
        #     orig_img.append(image)
        # print(image)
        blurred = apply_motion_blur(image)
        x_processed_train.append(blurred)
        # print(angle)
        num += 1
        print(num)
    x_processed_train = np.concatenate(x_processed_train, axis=0)

    # orig_img = np.array(orig_img)
    # x_processed_train = x_processed_train - orig_img
    print(x_processed_train.shape)
    np.save('x_processed_train.npy', x_processed_train)
    np.save('y_train.npy', y_train)
    np.save('kernels.npy', all_kernels)
    return x_processed_train, y_train, all_kernels

if __name__ == "__main__":
    process_data()