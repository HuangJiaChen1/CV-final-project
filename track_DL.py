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
# Callback function to handle mouse clicks
patch = np.zeros((32,32,3))
orig_img = np.zeros((32,32,3))
def click_event(event, x, y, flags, param):
    global patch
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate the coordinates of the top left corner of the patch
        x1 = max(x - 16, 0)
        y1 = max(y - 16, 0)

        # Adjust patch size if near the edges
        patch_width = min(32, img.shape[1] - x1)
        patch_height = min(32, img.shape[0] - y1)

        # Extract the patch
        patch = img[y1:y1+patch_height, x1:x1+patch_width]

        # Show the patch in a new window
        cv2.imshow("Patch", patch)

def click_event2(event, x, y, flags, param):
    global orig_img
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate the coordinates of the top left corner of the patch
        x1 = max(x - 16, 0)
        y1 = max(y - 16, 0)

        # Adjust patch size if near the edges
        patch_width = min(32, img2.shape[1] - x1)
        patch_height = min(32, img2.shape[0] - y1)

        # Extract the patch
        orig_img = img2[y1:y1+patch_height, x1:x1+patch_width]

        # Show the patch in a new window
        cv2.imshow("original", orig_img)
# Load an image
img = cv2.imread('./frame_lk/frame_37.jpg')  # Update the path to your image

# Display the image in a window
cv2.imshow("Image", img)
# Set the mouse callback function to detect clicks on the image window
cv2.setMouseCallback("Image", click_event)

# Wait indefinitely until you press any key while focused on one of the OpenCV windows
cv2.waitKey(0)
print(patch.shape)
# Destroy all OpenCV windows
cv2.destroyAllWindows()


img2 = cv2.imread('./frame_lk/frame_1.jpg')
cv2.imshow("Original",img2)
cv2.setMouseCallback('Original',click_event2)
cv2.waitKey(0)
print(orig_img.shape)
cv2.destroyAllWindows()

input = np.concatenate((patch,orig_img),axis=-1)
input_batch = np.expand_dims(input, axis=0)
out = model.predict(input_batch)
print(out)
out_class = np.argmax(out)
plt.figure(figsize=(10, 5))
plt.imshow(all_kernels[out_class])
plt.title('Predicted Kernel')
plt.axis('off')
plt.show()