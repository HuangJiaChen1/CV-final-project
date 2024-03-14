import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt


# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_built_with_cuda())
(x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

print(x_train[0].shape)


def generate_motion_blur_kernel(size=5, angle=0, speed=1):
    """
    Generates a motion blur kernel given a size, an angle, and a speed factor.
    The 'speed' parameter controls the density of the kernel to simulate different speeds.
    """
    # Basic kernel based on size
    kernel = np.zeros((size, size))

    # Adjusting the kernel based on speed
    if speed <= 1:
        # For slower speeds, concentrate weight in the center
        kernel[int((size - 1) / 2), :] = np.linspace(1 - speed, 1, num=size)
    else:
        # For higher speeds, spread weight towards the edges
        spread = int(np.clip(speed, 1, size // 2))
        center = int((size - 1) / 2)
        kernel[center - spread:center + spread + 1, :] = np.linspace(0.1, 1, num=2 * spread + 1).reshape(-1, 1)

    kernel = kernel / kernel.sum()  # Normalize the kernel

    # Rotating the kernel to the specified angle
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))

    return kernel


def apply_motion_blur(image_path, kernel_size=5, speed=1):
    """
    Applies a motion blur effect to an image using a randomly generated motion blur kernel
    with a specified speed to adjust the density of the kernel.
    """
    # Load the image
    img = image_path
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    # Generate a random angle for the motion blur kernel
    angle = np.random.uniform(0, 360)

    # Generate the motion blur kernel with the given speed
    kernel = generate_motion_blur_kernel(size=kernel_size, angle=angle, speed=speed)

    # Apply the motion blur kernel to the image
    blurred_img = cv2.filter2D(img, -1, kernel)

    # Display the original and blurred images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_img)
    plt.title('Motion Blurred Image')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(kernel)
    plt.title('Kernel')
    plt.axis('off')
    plt.show()

    # Return the motion blur kernel
    return kernel


y_train = np.zeros_like(x_train)
print(y_train.shape)
y_test = np.zeros_like(x_test)
# random kernel size and random angle
for i in range(x_train.shape[0]):
    image = x_train[0]
    # print(image)
    speed = int(np.random.uniform(0,8))
    k = int(np.random.uniform(3, 20))
    kernel = apply_motion_blur(image, kernel_size= k,speed= 0)
    print("Motion Blur Kernel:")
    print(kernel)

    # y_train[i] = kernel

