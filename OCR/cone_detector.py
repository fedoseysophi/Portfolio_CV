# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from skimage.measure import label, regionprops, regionprops_table
from skimage import feature

def hog_finder(img):
    # calculation of histogram of oriented gradients
    fd = feature.hog(img, orientations=18, pixels_per_cell=(5, 5), cells_per_block=(4, 4), channel_axis=2)
    return fd

def lbp_finder(img):
    # calculation of local binary patterns
    lbp = feature.local_binary_pattern(img[:,:,0], 9, 1, method="uniform")
    # forming the histogram to use as a feature vector
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 9 + 3), range=(0, 9 + 2))
    # histogram normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

vect1 = np.load('vect1.npy')
vect2 = np.load('vect2.npy')
vect3 = np.load('vect3.npy')

def finder(image):
    test_mask = (image[:,:,0]>140)&(image[:,:,0]<200)&(image[:,:,1]>95)&(image[:,:,1]<130)&(image[:,:,2]>100)&(image[:,:,2]<160)
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.dilate(test_mask.astype(np.uint8),kernel,iterations = 10)
    gray = cv2.erode(gray,kernel,iterations = 12)
    gray = cv2.dilate(gray,kernel,iterations = 9)
    label_im = label(gray)
    regions = regionprops(label_im)
    bbox = []
    for x in regions:
        area = x.area
        if (area>100) and (area<2000):
            bbox.append(x.bbox)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.imshow(image)
    count = 0
    for box in bbox:
        object = image[box[0]:box[2],box[1]:box[3]]
        object = cv2.resize(object, dsize=(45,45), interpolation=cv2.INTER_CUBIC)
        hog = hog_finder(object)
        histlbp = lbp_finder(object)
        vect = np.hstack((histlbp,hog))
        result = np.sqrt(np.square(vect - vect1).sum())
        result = result + np.sqrt(np.square(vect - vect2).sum())
        result = result + np.sqrt(np.square(vect - vect3).sum())
        if result < 30:
            rect = patches.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],
            linewidth=1, edgecolor='r', facecolor='none')
            count = count + 1
            ax.add_patch(rect)
    plt.show()
    print('Number of cones =',count)
    # add necessary lines to form the information returned by the function
    # assign the variable new_image the obtained image with bounding boxes
    new_image = np.array(fig.canvas.renderer.buffer_rgba())
    return count, new_image # after return list everything that the function should return

image = plt.imread('frame_7.jpg')
count, new_image = finder(image)
image = plt.imread('frame_8.jpg')
count, new_image = finder(image)
image = plt.imread('frame_9.jpg')
count, new_image = finder(image)
image = plt.imread('frame_10.jpg')
count, new_image = finder(image)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from skimage.measure import label, regionprops
from skimage import feature

# Calculation of HOG features
def hog_finder(img):
    fd = feature.hog(img, orientations=18, pixels_per_cell=(5, 5), cells_per_block=(4, 4), channel_axis=2)
    return fd

# Calculation of LBP features
def lbp_finder(img):
    lbp = feature.local_binary_pattern(img[:, :, 0], 9, 1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 9 + 3), range=(0, 9 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Reference vectors
vect1 = np.load('vect1.npy')
vect2 = np.load('vect2.npy')
vect3 = np.load('vect3.npy')

# Object detection
def finder(image):
    test_mask = (image[:, :, 0] > 140) & (image[:, :, 0] < 200) & \
                (image[:, :, 1] > 95) & (image[:, :, 1] < 130) & \
                (image[:, :, 2] > 100) & (image[:, :, 2] < 160)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(test_mask.astype(np.uint8), kernel, iterations=10)
    gray = cv2.erode(gray, kernel, iterations=12)
    gray = cv2.dilate(gray, kernel, iterations=9)
    label_im = label(gray)
    regions = regionprops(label_im)

    bbox = []
    for x in regions:
        area = x.area
        if 100 < area < 2000:  # Area filtering
            bbox.append(x.bbox)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    count = 0

    for box in bbox:
        obj = image[box[0]:box[2], box[1]:box[3]]
        obj = cv2.resize(obj, dsize=(45, 45), interpolation=cv2.INTER_CUBIC)
        hog = hog_finder(obj)
        histlbp = lbp_finder(obj)
        vect = np.hstack((histlbp, hog))
        result = np.sqrt(np.square(vect - vect1).sum())
        result += np.sqrt(np.square(vect - vect2).sum())
        result += np.sqrt(np.square(vect - vect3).sum())
        if result < 30:  # Classification threshold
            rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=2, edgecolor='g', facecolor='none')  # Green box
            count += 1
            ax.add_patch(rect)

    plt.show()
    print('Number of parking signs =', count)
    return count, fig

# Example usage
images = ['frame_7.jpg', 'frame_8.jpg', 'frame_9.jpg', 'frame_10.jpg']
for img_path in images:
    image = plt.imread(img_path)
    count, result_fig = finder(image)
    result_fig.savefig(f'result_{img_path}')  # Saving the processed image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from skimage.measure import label, regionprops
from skimage import feature

# Calculation of HOG features
def hog_finder(img):
    fd = feature.hog(img, orientations=18, pixels_per_cell=(5, 5), cells_per_block=(4, 4), channel_axis=2)
    return fd

# Calculation of LBP features
def lbp_finder(img):
    lbp = feature.local_binary_pattern(img[:, :, 0], 9, 1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 9 + 3), range=(0, 9 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Object detection
def finder(image, color_ranges):
    test_mask = np.zeros_like(image[:, :, 0], dtype=bool)
    for color_range in color_ranges:
        mask = (image[:, :, 0] >= color_range[0][0]) & (image[:, :, 0] <= color_range[0][1]) & \
               (image[:, :, 1] >= color_range[1][0]) & (image[:, :, 1] <= color_range[1][1]) & \
               (image[:, :, 2] >= color_range[2][0]) & (image[:, :, 2] <= color_range[2][1])
        test_mask |= mask

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(test_mask.astype(np.uint8), kernel, iterations=10)
    gray = cv2.erode(gray, kernel, iterations=12)
    gray = cv2.dilate(gray, kernel, iterations=9)
    label_im = label(gray)
    regions = regionprops(label_im)

    bbox = []
    for x in regions:
        area = x.area
        if 100 < area < 2000:  # Area filtering
            bbox.append(x.bbox)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    count = 0

    for box in bbox:
        obj = image[box[0]:box[2], box[1]:box[3]]
        obj = cv2.resize(obj, dsize=(45, 45), interpolation=cv2.INTER_CUBIC)
        hog = hog_finder(obj)
        histlbp = lbp_finder(obj)
        vect = np.hstack((histlbp, hog))
        # Use reference vectors if available
        # result = np.sqrt(np.square(vect - vect1).sum())
        # Add comparison with vectors if required
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=2, edgecolor='g', facecolor='none')  # Green box
        count += 1
        ax.add_patch(rect)

    plt.show()
    print('Number of parking signs =', count)
    return count, fig

# Example usage
images = ['/mnt/data/template (2).jpg', '/mnt/data/SP_1 (2).png']
color_ranges = [
    # Color ranges (min, max) for each channel (BGR)
    [(140, 200), (95, 130), (100, 160)],  # Example: range for parking signs
]

for img_path in images:
    image = plt.imread(img_path)
    count, result_fig = finder(image, color_ranges)
    result_fig.savefig(f'result_{img_path.split('/')[-1]}')  # Saving the processed image
