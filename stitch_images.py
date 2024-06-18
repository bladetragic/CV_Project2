# Brandon Bowles
# CSE4310 - 001
# Assignment 2

# Was not able to get my implementation of Ransac function working properly, So it is commented out. 
# Used library Ransac so I could move on to at least try to implement image stitching.
# Image Stitching is functioning to some degree. 

import matplotlib.pyplot as plt
import sys
import numpy as np
from skimage import io
import math
import PIL.Image as Image
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import mean_squared_error
from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform, warp
import scipy.ndimage as ndi
from skimage.measure import ransac
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import match_descriptors, plot_matches, SIFT
from scipy.spatial.distance import cdist
from keypoint_matching import bf_match_descriptors

# def bf_match_descriptors(descriptors1, descriptors2, max_ratio = 0.1):
#     # Calculate Euclidean distance between descriptors
#     distances = cdist(descriptors1, descriptors2)
#     #distances = np.linalg.norm(descriptors1-descriptors2)
#     #print(descriptors1.shape)
#     #print(descriptors2.shape)
#     #print(distances.shape)
#     #print(distances)

    
#     # Find the nearest neighbor for each descriptor
#     matches = np.argmin(distances, axis=1)
#     print(matches.shape)
#     print(matches)
#     # for i in matches:
#     #     print (distances[i][matches[i]])
    
#     # Compute the ratio of the nearest neighbor distance to the second nearest neighbor distance
#     min_distances = np.min(distances, axis=1)
#     second_min_distances = np.partition(distances, 1, axis=1)[:, 1]
#     ratio = min_distances / second_min_distances
    
#     # Filter matches based on threshold
#     valid_matches = ratio < max_ratio
    
#     # Create array of matched pairs
#     matched_pairs = np.column_stack((np.arange(len(matches))[valid_matches], matches[valid_matches]))
    
#     return matched_pairs

def compute_affine_transform(src_points, dst_points):
    # Ensure the lists of points are the same length
    assert len(src_points) == len(dst_points)

    # Create matrix A
    A = np.zeros((2*len(src_points), 6))
    for i in range(len(src_points)):
        A[2*i] = [-src_points[i][0], -src_points[i][1], -1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, -src_points[i][0], -src_points[i][1], -1]
        

    # Create matrix B
    B = np.array(dst_points).reshape(2*len(dst_points), 1)

    # Solve for the affine transformation matrix
    affine_matrix = np.linalg.lstsq(A, B, rcond=None)[0]

    # Reshape the solution into a 3x3 matrix
    affine_matrix = np.vstack((affine_matrix.reshape(2, 3), [0, 0, 1]))

    return affine_matrix

def compute_projective_transform(src_points, dst_points):
    # Ensure the lists of points are the same length
    assert len(src_points) == len(dst_points)

    # Create matrix A
    A = np.zeros((2*len(src_points), 9))
    for i in range(len(src_points)):
        X, Y = src_points[i]
        x, y = dst_points[i]
        A[2*i] = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x]
        A[2*i+1] = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y]

    # Solve for the projective transformation matrix
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    return H

# def ransac(src_points, dst_points, num_iterations, min_samples, threshold):
#     best_model = None
#     best_inliers = np.array([])  # Initialize best_inliers as an empty numpy array
#     best_error = np.inf

#     for _ in range(num_iterations):
#         # Randomly select min_samples
#         indices = np.random.choice(len(src_points), min_samples, replace=False)
#         sample_src = src_points[indices]
#         sample_dst = dst_points[indices]

#         # Compute the transformation matrix using the selected samples
#         H = compute_projective_transform(sample_src, sample_dst)

#         # Apply the transformation to all source points
#         transformed_src = np.dot(H, np.vstack((src_points.T, np.ones(len(src_points))))).T
#         transformed_src /= transformed_src[:, 2][:, np.newaxis]

#         # Convert transformed_src from homogeneous to Cartesian coordinates
#         transformed_src = transformed_src[:, :2]

#         # Compute the error (distance) between the transformed source points and the destination points
#         error = np.sqrt(mean_squared_error(transformed_src, dst_points))

#         # Determine the inliers (points where the error is less than the threshold)
#         inliers = error < threshold

#         # If the number of inliers is greater than the number of best inliers found so far, update the best model
#         if np.sum(inliers) > np.sum(best_inliers):
#             best_model = H
#             best_inliers = inliers
#             best_error = error

#     return best_model, best_inliers, best_error


# Get file name for source image from command line argument 1
img1 = io.imread(sys.argv[1])
# img1 = np.asarray(Image.open(sys.argv[1]))

#Get file name for destination image from command line argument 2
img2 = io.imread(sys.argv[2])
# img2 = np.asarray(Image.open(sys.argv[2]))


if img1.shape[2] == 4:
    img1 = rgba2rgb(img1)
if img2.shape[2] == 4:
    img2 = rgba2rgb(img2)

# dst_img_rgb = np.asarray(Image.open(sys.argv[1]))
# src_img_rgb = np.asarray(Image.open(sys.argv[2]))

# if dst_img_rgb.shape[2] == 4:
#     dst_img_rgb = rgba2rgb(dst_img_rgb)
# if src_img_rgb.shape[2] == 4:
#     src_img_rgb = rgba2rgb(src_img_rgb)

# dst_img = rgb2gray(dst_img_rgb)
# src_img = rgb2gray(src_img_rgb)


# Extract keypoints from image using SIFT
img1_gray = rgb2gray(img1)
sift = SIFT()
sift.detect_and_extract(img1_gray)
#sift.detect_and_extract(dst_img)
keypoints1 = sift.keypoints
descriptors1 = sift.descriptors

# Extract keypoints from image using SIFT
img2_gray = rgb2gray(img2)
sift.detect_and_extract(img2_gray)
keypoints2 = sift.keypoints
descriptors2 = sift.descriptors

# Match descriptors between img1 and img2
matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.8)
#matches12 = bf_match_descriptors(descriptors1, descriptors2)

dst_coordinates = keypoints1[matches12[:, 0]]
src_coordinates = keypoints2[matches12[:, 1]]
#print(matches12[:,0], matches12[:,1])

affine_matrix = compute_affine_transform(dst_coordinates[:, ::-1], src_coordinates[:, ::-1])
projective_matrix = compute_projective_transform(dst_coordinates[:, ::-1], src_coordinates[:, ::-1])

print(affine_matrix)
print("---------------------------------------------")
print(projective_matrix)

model12, inliers12 = ransac((dst_coordinates[:, ::-1], src_coordinates[:, ::-1]),  # source and destination coordinates
                            AffineTransform,  # the model class
                            min_samples=3,  # minimum number of coordinates to fit the model
                            residual_threshold=2,  # maximum distance for a data point to be considered an inlier
                            max_trials=100)  # maximum number of iterations

#model12, inliers12 = ransac(keypoints1[matches12[:, 0]][:, ::-1], keypoints2[matches12[:, 1]][:, ::-1], 100, 3, 2)
#best_model, best_inliers, best_error = ransac(keypoints1[matches12[:, 0]][:, ::-1], keypoints2[matches12[:, 1]][:, ::-1], 100, 3, 2)
outliers12 = inliers12 == False

# Display inlier and outlier matches
fig, ax = plt.subplots()
plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12[inliers12], matches_color='g')
plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12[outliers12], matches_color='r')
ax.axis('off')
plt.show()

sk_M, sk_best = ransac((src_coordinates[:, ::-1], dst_coordinates[:, ::-1]), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)
print(sk_M)

print(np.count_nonzero(sk_best))
src_best = keypoints2[matches12[sk_best, 1]][:, ::-1]
dst_best = keypoints1[matches12[sk_best, 0]][:, ::-1]

# fig = plt.figure(figsize=(8, 4))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# ax1.imshow(img2)
# ax2.imshow(img1)

# for i in range(src_best.shape[0]):
#     coordB = [dst_best[i, 0], dst_best[i, 1]]
#     coordA = [src_best[i, 0], src_best[i, 1]]
#     con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
#                           axesA=ax2, axesB=ax1, color="red")
#     ax2.add_artist(con)
#     ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
#     ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')

# Transform the corners of img1 by the inverse of the best fit model
rows, cols = img1_gray.shape
corners = np.array([
    [0, 0],
    [cols, 0],
    [0, rows],
    [cols, rows]
])

corners_proj = sk_M(corners)
all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)
output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1]).astype(int)
print(output_shape)

offset = SimilarityTransform(translation=-corner_min)
dst_warped = warp(img2, offset.inverse, output_shape=output_shape)

tf_img = warp(img1, (sk_M + offset).inverse, output_shape=output_shape)

# Combine the images
foreground_pixels = tf_img[tf_img > 0]
dst_warped[tf_img > 0] = tf_img[tf_img > 0]

# Plot the result
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(dst_warped)
plt.show()

