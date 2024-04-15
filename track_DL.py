import cv2
import numpy as np
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
max_iter = 20
def track(iter,x,y,w,h,last_dim,p):
    while iter <= max_iter:
        X = np.arange(x + int(p[0]), x + int(p[0]) + w, dtype=np.float32)
        Y = np.arange(y + int(p[1]), y + int(p[1]) + h, dtype=np.float32)
        X, Y = np.meshgrid(X, Y)
        warp = np.array([[1, 0], [0, 1]])
        I = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
        dim = np.float32(I) - np.float32(template)

        if np.linalg.norm(last_dim - dim) <= 0.001:
            break
        last_dim = dim

        dy, dx = np.gradient(np.float32(I))
        A = np.dot(np.hstack((dx.reshape(-1, 1), dy.reshape(-1, 1))),warp)
        b = -dim.reshape(-1, 1)
        u = np.dot(np.linalg.pinv(A), b)
        # print("u:",u.shape)
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


def select_roi(image):
    r = cv2.selectROI("Select ROI", image, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return r


def create_particles(x, y, w, h, num_particles=1000):
    particles = np.empty((num_particles, 2))
    particles[:, 0] = x + (np.random.rand(num_particles) - 0.5) * w
    particles[:, 1] = y + (np.random.rand(num_particles) - 0.5) * h
    return particles

def apply_motion_vectors_to_particles(particles, motion_vectors, w, h, patch_size=32):
    n_patches_x = w // patch_size
    n_patches_y = h // patch_size
    n = n_patches_x * n_patches_y  # Total number of patches

    # Ensure motion_vectors shape is (2, n)
    assert motion_vectors.shape == (2, n), "Motion vectors shape mismatch"

    # Adjust each particle's position based on the average motion vector of all patches
    avg_motion_vector = np.mean(motion_vectors, axis=1)  # Shape (2,)
    particles += avg_motion_vector.reshape(1, 2)  # Broadcast addition

    return particles
# def particle_filter_update(particles, frame_gray, template, x, y, w, h, num_particles=100, sigma=10.0):
#     particles = apply_motion_vectors_to_particles(particles, motion_vectors, w, h)
#     weights = np.zeros(num_particles)
#     for i, particle in enumerate(particles):
#         px, py = int(particle[0]), int(particle[1])
#         try:
#             particle_roi = frame_gray[py:py + h, px:px + w]
#             if particle_roi.shape[0] != h or particle_roi.shape[1] != w:
#                 weights[i] = 0.1
#             else:
#                 error = np.sum((template.astype(float) - particle_roi.astype(float)) ** 2)
#                 weights[i] = np.exp(-error / (2 * (sigma ** 2)))
#         except IndexError:
#             weights[i] = 0.1
#     weights += 1.e-10  # Avoid division by zero
#     weights /= weights.sum()
#
#     indices = np.random.choice(num_particles, num_particles, p=weights)
#     particles[:] = particles[indices]
#     mean_particle = np.mean(particles, axis=0)
#     return particles, mean_particle


# Initialize video capture
cam = cv2.VideoCapture("vid.mp4")
ret_val, first_frame = cam.read()
if not ret_val:
    print("Failed to capture from camera. Exiting.")
    cam.release()
    exit()

# Select ROI
r = select_roi(first_frame)
x, y, w, h = r
X = np.arange(x, x + w, dtype=np.float32)
Y = np.arange(y, y + h, dtype=np.float32)
X, Y = np.meshgrid(X, Y)

template_rgb = cv2.remap(first_frame, X, Y, cv2.INTER_LINEAR)
template_patches = img2patch(template_rgb)
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
template = first_frame_gray[y:y + h, x:x + w]
num_particles = 100
particles = create_particles(x, y, w, h, num_particles)
n = 0
mean_particle = np.array([x + w / 2, y + h / 2])
last_dim = np.zeros_like(template)
p = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
while True:
    ret_val, frame = cam.read()
    if not ret_val:
        break
    n += 1
    '''
    Crop the current ROI here
    '''
    # mean_x, mean_y = int(mean_particle[0]), int(mean_particle[1])
    # start_x = max(mean_x - w // 2, 0)
    # start_y = max(mean_y - h // 2, 0)
    # end_x = min(start_x + w, frame.shape[1])
    # end_y = min(start_y + h, frame.shape[0])
    # I = frame[start_y:end_y, start_x:end_x]
    X = np.arange(x + int(p[0]), x + int(p[0]) + w, dtype=np.float32)
    Y = np.arange(y + int(p[1]), y + int(p[1]) + h, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    warp = np.array([[1, 0], [0, 1]])
    I = cv2.remap(frame, X, Y, cv2.INTER_LINEAR)
    # cv2.imshow('image',I)
    # cv2.waitKey(0)
    print(I.shape)
    frame_patches = img2patch(I)
    print(len(frame_patches))
    motion_vectors1 = np.empty((2,len(frame_patches)))
    motion_vectors2 = np.empty((2, len(frame_patches)))
    for i in range(len(frame_patches)):
        input = np.concatenate((frame_patches[i], template_patches[i]), axis=-1)
        input_batch = np.expand_dims(input, axis=0)
        out = model.predict(input_batch)
        out_class = np.argmax(out)
        size, angle = index_to_size_angle(out_class)
        v1 = size * np.cos(np.radians(angle))
        v2 = size * np.sin(np.radians(angle))
        motion_vectors1[0,i] = v1
        motion_vectors1[1,i] = v2
        motion_vectors2[0, i] = -v1
        motion_vectors2[1, i] = -v2
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    u1 = np.mean(motion_vectors1, axis=1)
    u2 = np.mean(motion_vectors2, axis=1)
    # cv2.rectangle(frame, (x,y), (x  + w, y  + h), (0, 255, 0), 2)
    print(u1)
    candidate_p1 = p.copy()
    candidate_p2 = p.copy()
    candidate_p3 = p.copy()
    candidate_p1 += np.array(u1).astype(np.float32).reshape(-1, 1)
    candidate_p3 +=np.array(u2).astype(np.float32).reshape(-1, 1)
    iter = 0
    candidate_p1 = track(iter, x, y, w, h, last_dim, candidate_p1)
    candidate_p2 = track(iter, x, y, w, h, last_dim, candidate_p2)
    candidate_p3 = track(iter, x, y, w, h, last_dim, candidate_p3)
    X = np.arange(x + int(candidate_p1[0]), x + int(candidate_p1[0]) + w, dtype=np.float32)
    Y = np.arange(y + int(candidate_p1[1]), y + int(candidate_p1[1]) + h, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    candidate_img1 = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
    X = np.arange(x + int(candidate_p2[0]), x + int(candidate_p2[0]) + w, dtype=np.float32)
    Y = np.arange(y + int(candidate_p2[1]), y + int(candidate_p2[1]) + h, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    candidate_img2 = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
    X = np.arange(x + int(candidate_p3[0]), x + int(candidate_p3[0]) + w, dtype=np.float32)
    Y = np.arange(y + int(candidate_p3[1]), y + int(candidate_p3[1]) + h, dtype=np.float32)
    X, Y = np.meshgrid(X, Y)
    candidate_img3 = cv2.remap(frame_gray, X, Y, cv2.INTER_LINEAR)
    sqr_diff1 = (candidate_img1-template) **2
    sqr_diff2 = (candidate_img2-template) **2
    sqr_diff3 = (candidate_img3 - template) ** 2
    ssd1 = np.sum(sqr_diff1)
    ssd2 = np.sum(sqr_diff2)
    ssd3 = np.sum(sqr_diff3)
    if ssd1 == min(ssd1,ssd2,ssd3):
        p = candidate_p1
    elif ssd2 == min(ssd1,ssd2,ssd3):
        p = candidate_p2
    elif ssd3 == min(ssd1,ssd2,ssd3):
        p = candidate_p3
    cv2.rectangle(frame, (x + int(p[0]), y + int(p[1])), (x + int(p[0]) + w, y + int(p[1]) + h), (255, 0, 0), 2)
    # x += int(u[0])
    # y += int(u[1])

    # # Update particles
    # particles, mean_particle = particle_filter_update(particles, frame_gray, template, x, y, w, h, num_particles)
    #
    # # Visualization: draw particles
    # for particle in particles:
    #     cv2.circle(frame, (int(particle[0]), int(particle[1])), 2, (255, 0, 0), -1)
    #
    # # Draw estimated object position
    # cv2.rectangle(frame, (int(mean_particle[0] - w / 2), int(mean_particle[1] - h / 2)),
    #               (int(mean_particle[0] + w / 2), int(mean_particle[1] + h / 2)), (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    cv2.waitKey(0)
    cv2.imwrite(f'./frame_dl/frame_{n}.jpg', frame)
    k = cv2.waitKey(30)
    if k == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
