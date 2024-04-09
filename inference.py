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
    model.add(layers.MaxPool2D((2, 2), strides=2))

    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    #
    # model.add(layers.Conv2D(512, (5,5), activation= 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(units= len(all_kernels), activation='softmax'))
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
def index_to_size_angle(index):
    if index == 0:
        return (1,0)
    else:
        index -=1
    num_sizes = 12
    angle_step = 10
    size_step = 2
    size_start = 2

    angle_index = index // num_sizes
    size_index = index % num_sizes

    angle = angle_index * angle_step
    size = size_index * size_step + size_start

    if angle > 170 or size > 25:
        print(size,angle)
        print("Index out of range")
        return "Index out of range"
    else:
        return (size, angle)

def generate_motion_blur_kernel(size, angle):
    kernel = np.zeros((30, 30))
    array = np.zeros((30,1), dtype=int)
    start_index = (30 - size) // 2
    array[start_index:start_index + size, :] = 1
    kernel[14,:] =array.flatten()
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((15,15), angle, 1.0), (30, 30))
    kernel = kernel / np.sum(kernel)
    return kernel
model = create_model()
model.load_weights("model")
x_test = np.concatenate((training.x_train[0:100],training.x_train[272:373]),axis= 0)
print(x_test[0])
predictions = model.predict(x_test)
n = 0
correct = 0
for i, prediction in enumerate(predictions):
    print(f"Predicted parameters for sample {i}: {prediction}")
    # image = x_test[i]
    # k = int(prediction[0])
    # angle = prediction[1]
    # kernel = training.generate_motion_blur_kernel(k,angle)
    # blurred_img = cv2.filter2D(image, -1, kernel)
    out_class = np.argmax(prediction)
    print(out_class)
    size,angle = index_to_size_angle(out_class)
    k = generate_motion_blur_kernel(size,angle)
    # print(prediction[out_class])
    # print(training.y_train[i])
    print(prediction[training.y_train[i][0]])
    n += 1
    if prediction[out_class] == prediction[training.y_train[i][0]]:
        correct +=1
        print("correct")
    else:
        print("wrong")
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(all_kernels[out_class])
    plt.title('Predicted Kernel')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(all_kernels[training.y_train[i][0]])
    plt.title("Ground Truth Kernel")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(k)
    plt.title("Kernel From Index")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(x_test[i][:,:,:3])
    plt.title("train image")
    plt.axis('off')
    plt.show()
print('accuracy: ', correct/n)