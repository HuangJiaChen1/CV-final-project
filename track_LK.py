import cv2
import numpy as np
from matplotlib import pyplot as plt

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

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
template = cv2.remap(first_frame_gray, X, Y, cv2.INTER_LINEAR)

p = np.array([0, 0]).astype(np.float32).reshape(-1, 1)
max_iter = 100
last_dim = np.zeros_like(template)
n = 0
while True:
    ret_val, frame = cam.read()
    if not ret_val:
        break
    n += 1
    # Visualization
    # X = np.arange(x, x + w, dtype=np.float32) + p[0]
    # Y = np.arange(y, y + h, dtype=np.float32) + p[1]
    # X, Y = np.meshgrid(X, Y)
    # warp = np.array([[1, 0], [0, 1]])
    # I = cv2.remap(frame, X, Y, cv2.INTER_LINEAR)
    # cv2.imshow('image', I)
    # cv2.waitKey(0)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    u = np.array([0, 0])
    iter = 0
    p = track(iter,x,y,w,h,last_dim,p)
    cv2.rectangle(frame, (x + int(p[0]), y + int(p[1])), (x + int(p[0]) + w, y + int(p[1]) + h), (0, 255, 0), 2)
    cv2.imshow('Tracking', frame)
    cv2.imwrite(f'./frame_lk/frame_{n}.jpg', frame)
    k = cv2.waitKey(1)
    if k == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
