# Brandon Bowles

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import SIFT
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from tqdm import tqdm


# TODO: Create feature processing functions for SIFT and HOG
def extract_sift_features(images, targets):
    
    # Extract SIFT features per class
    sift = SIFT()
    sift_features = []
    y_features = []

    #print("Images shape: ", images.shape)

    for idx in tqdm(range(images.shape[0]), desc="Processing images"):
            try:
                sift.detect_and_extract(images[idx])
                descriptors1 = sift.descriptors
                sift_features.append(descriptors1)
                y_features.append(targets[idx]) # Only stores the label if the SIFT features are successfully extracted`
            except:
                pass
    
   
    # fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    # for i, ax in enumerate(axes):
    #     ax.imshow(images[i], cmap='gray')
    #     ax.axis('off')
    # plt.show()

    return sift_features, y_features

def build_histograms(sift_features):
    # Build a histogram of the cluster centers for each image using the features already extracted
    image_histograms = []

    for feature in tqdm(sift_features, desc="Building histograms"):
        # Predict the closest cluster for each feature
        clusters = kmeans.predict(feature)
        # Build a histogram of the clusters
        histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
        image_histograms.append(histogram)

    return image_histograms

if __name__ == "__main__":
    
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    X_train = data['X_train'].astype(np.uint8)
    y_train = data['y_train']
    X_test = data['X_test'].astype(np.uint8)
    y_test = data['y_test']

    X_train_rgb = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_train_gray = rgb2gray(X_train_rgb)

    X_test_rgb = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test_gray = rgb2gray(X_test_rgb)
    

    # TODO: Extract features from the training data
    
    # Process training data
    sift_features_train, y_features_train = extract_sift_features(X_train_gray, y_train)

    # TODO: Extract features from the testing data
    
    # Process testing data
    sift_features_test, y_features_test = extract_sift_features(X_test_gray, y_test)

    #Calculate number of features extracted
    feature_number = len(sift_features_test) + len(sift_features_train)
    print(f"Number of features found: {feature_number}")
   
    # Convert the list of SIFT features to a numpy array
    sift_features_train_np = np.concatenate(sift_features_train)
    sift_features_test_np = np.concatenate(sift_features_test)
    sift_features_np = np.concatenate((sift_features_train_np, sift_features_test_np))

    
    # Create a KMeans model to cluster the SIFT features
    vocab_size = 50
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)

    # Fit the KMeans model to the SIFT features
    kmeans.fit(sift_features_np)

    
    #Build histograms for features
    sift_training_histogram = build_histograms(sift_features_train)

    sift_testing_histogram = build_histograms(sift_features_test)

    # Convert the list of histograms to a numpy array
    sift_training_histograms_np = np.array(sift_training_histogram)
    sift_testing_histograms_np = np.array(sift_testing_histogram)

    # Convert y features to a numpy array
    np.array(y_features_train, dtype=int)
    np.array(y_features_test, dtype=int)

    
    # Save data to a npz file in a dictionary-like format 
    # Note that it is a dictionary of numpy arrays
    sift_data = {
        "X_train": sift_training_histograms_np,
        "X_test": sift_testing_histograms_np,
        "y_train": y_features_train,
        "y_test": y_features_test
    }

    # TODO: Save the extracted features to a file
    np.savez('sift_features.npz', **sift_data)
    
    
