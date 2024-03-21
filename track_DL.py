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

    return model

model = create_model()
model.load_weights("model")
model_outputs = []
def select_roi(image):
    r = cv2.selectROI("Select ROI", image, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return r
def track(iter,x,y,w,h,last_dim,p):
    while iter <= max_iter:
        X = np.arange(x, x + w, dtype=np.float32) + p[0]
        Y = np.arange(y, y + h, dtype=np.float32) + p[1]
        X, Y = np.meshgrid(X, Y)
        warp = np.array([[1,0],[0,1]])
        I = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
        dim = np.float32(I) - np.float32(template)

        if np.linalg.norm(last_dim - dim) <= 0.001:
            break
        last_dim = dim

        dy, dx = np.gradient(np.float32(I))
        A = np.dot(np.hstack((dx.reshape(-1, 1), dy.reshape(-1, 1))),warp)
        b = -dim.reshape(-1, 1)
        u = np.dot(np.linalg.pinv(A), b)
        p += u
        iter += 1
    return p

def img2patch(image):
    """
    Converts an image into an array of (32x32x3) patches.

    Parameters:
    - image: A numpy array of shape (m, n, 3).

    Returns:
    - A list of (32, 32, 3) patches extracted from the image.
    """
    image = np.asarray(image)
    m, n, c = image.shape
    patches = []
    if c != 3:
        raise ValueError("Image must have 3 channels.")
    vertical_patches = m // 32
    horizontal_patches = n // 32

    for i in range(vertical_patches):
        for j in range(horizontal_patches):
            patch = image[i*32:(i+1)*32, j*32:(j+1)*32, :]
            patches.append(patch)

    return patches


def most_frequent_element(arr):
    # Create a dictionary to count occurrences of each element
    count_dict = {}
    for element in arr:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1

    # Find the element with the maximum count
    max_count = 0
    most_frequent = arr[0]  # Initialize with the first element
    for element, count in count_dict.items():
        if count > max_count:
            max_count = count
            most_frequent = element

    return most_frequent

def index_to_size_angle(index):
    if index == 0:
        return (1,0)
    else:
        index -=1
    num_sizes = 12
    angle_step = 30
    size_step = 2
    size_start = 2

    angle_index = index // num_sizes
    size_index = index % num_sizes

    angle = angle_index * angle_step
    size = size_index * size_step + size_start

    if angle > 150 or size > 24:
        return "Index out of range"
    else:
        return (size, angle)


cam = cv2.VideoCapture("vid.mp4")


ret_val, first_frame = cam.read()
if not ret_val:
    print("Failed to capture from camera. Exiting.")
    cam.release()
    exit()

cv2.imshow('First Frame', first_frame)
r = select_roi(first_frame)
x, y, w, h = r
X = np.arange(x, x + w, dtype=np.float32)
Y = np.arange(y, y + h, dtype=np.float32)
X, Y = np.meshgrid(X, Y)

template_rgb = cv2.remap(first_frame, X, Y, cv2.INTER_LINEAR)
template_patches = img2patch(template_rgb)
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
template = cv2.remap(first_frame_gray, X, Y, cv2.INTER_LINEAR)

p = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
max_iter = 20
last_dim = np.zeros_like(template)
n = 0
while True:
    ret_val, frame = cam.read()
    if not ret_val:
        break
    n += 1
    frame_patches = img2patch(frame)
    #predict with CNN model
    for i in range(len(frame_patches)):
        input = np.concatenate((frame_patches, template_patches), axis=-1)
        input_batch = np.expand_dims(input, axis=0)
        out = model.predict(input_batch)
        out_class = np.argmax(out)
        model_outputs.append(out_class)
    k = most_frequent_element(model_outputs)
    size, angle = index_to_size_angle(k)
    v1 = size * np.cos(np.radians(angle))
    v2 = size * np.sin(np.radians(angle))
    v = np.array([v1,v2])
    p += v
    model_outputs = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    u = np.array([0, 0])
    iter = 0
    p = track(iter,x,y,w,h,last_dim,p)
    cv2.rectangle(frame, (x + int(p[0]), y + int(p[1])), (x + int(p[0]) + w, y + int(p[1]) + h), (0, 255, 0), 2)
    cv2.imshow('Tracking', frame)
    cv2.imwrite(f'./frame_dl/frame_{n}.jpg', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()

#
# print(out)
# out_class = np.argmax(out)
# plt.figure(figsize=(10, 5))
# plt.imshow(all_kernels[out_class])
# plt.title('Predicted Kernel')
# plt.axis('off')
# plt.show()