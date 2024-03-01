# Brandon Bowles
# CSE4310 - 001
# Assignment 2

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm


if __name__ == "__main__":
    
    # Load the pre-split data
    data = np.load("sift_features.npz", allow_pickle=True)

    # Load data from npz file
    # Data type: numpy arrrays
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # ------ Adjust frequency using TF-IDF ------
    
    # Create a TfidfTransformer
    tfidf = TfidfTransformer()

    # Fit the TfidfTransformer to the histogram data
    tfidf.fit(X_train)
    tfidf.fit(X_test)

    # Transform the histogram data using the trained TfidfTransformer
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # ------ Train an SVM classifier ------ 
    
    # Create an SVM model
    svm = LinearSVC(random_state=42)

    # Train the model
    svm.fit(X_train_tfidf, y_train)

    # Evaluate the model
    accuracy = svm.score(X_test_tfidf, y_test)
    print(f'SVM accuracy: {accuracy:.2f}')

   