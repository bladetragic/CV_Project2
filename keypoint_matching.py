# Brandon Bowles
# CSE4310 - 001
# Assignment 2

import sys
import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import match_descriptors, plot_matches, SIFT
from scipy.spatial.distance import cdist
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

def bf_match_descriptors(descriptors1, descriptors2, max_ratio = 0.7):
    # Calculate Euclidean distance between descriptors
    distances = cdist(descriptors1, descriptors2)
    #distances = np.linalg.norm(descriptors1-descriptors2)
    #print(descriptors1.shape)
    #print(descriptors2.shape)
    #print(distances.shape)
    #print(distances)

    
    # Find the nearest neighbor for each descriptor
    matches = np.argmin(distances, axis=1)
    print(matches.shape)
    print(matches)
    # for i in matches:
    #     print (distances[i][matches[i]])
    
    # Compute the ratio of the nearest neighbor distance to the second nearest neighbor distance
    min_distances = np.min(distances, axis=1)
    second_min_distances = np.partition(distances, 1, axis=1)[:, 1]
    ratio = min_distances / second_min_distances
    
    # Filter matches based on threshold
    valid_matches = ratio < max_ratio
    
    # Create array of matched pairs
    matched_pairs = np.column_stack((np.arange(len(matches))[valid_matches], matches[valid_matches]))
    
    return matched_pairs

def plot_keypoints(img1, img2, keypoints1, keypoints2, matches):
    # Select the points in img1 that match with img2 and vice versa
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')

    for i in range(src.shape[0]):
        coordB = [dst[i, 1], dst[i, 0]]
        coordA = [src[i, 1], src[i, 0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst[i, 1], dst[i, 0], 'ro')
        ax2.plot(src[i, 1], src[i, 0], 'ro')
    
#Get file name for source image from command line argument 1
img1 = io.imread(sys.argv[1])

#Get file name for destination image from command line argument 2
img2 = io.imread(sys.argv[2])


# Extract keypoints from image using SIFT
img1_gray = rgb2gray(img1)
sift = SIFT()
sift.detect_and_extract(img1_gray)
keypoints1 = sift.keypoints
descriptors1 = sift.descriptors

# Extract keypoints from image using SIFT
img2_gray = rgb2gray(img2)
sift.detect_and_extract(img2_gray)
keypoints2 = sift.keypoints
descriptors2 = sift.descriptors

# Match descriptors between img1 and img2
#matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.1)
matches12 = bf_match_descriptors(descriptors1, descriptors2)


# Display matches
#fig, ax = plt.subplots()
#plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
plot_keypoints(img1, img2, keypoints1, keypoints2, matches12)
#ax.axis('off')
plt.show()