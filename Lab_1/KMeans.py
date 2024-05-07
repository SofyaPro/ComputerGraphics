import numpy as np
import matplotlib.pylab as plt
from contourpy.util import data
import cv2 as cv
from scipy.spatial.distance import cdist


class MyKMeans:

    def __init__(self, clusters_number):

        self.clusters_number = clusters_number
        self.centroids = None
        self.X = None
        self.clusters = None

    def fit(self, X, max_iter=30):
        self.X = X

        self.centroids = np.random.uniform(low=np.min(X), high=np.max(X), size=(self.clusters_number, X.shape[1]))

        for _ in range(max_iter):

            diff = cdist(X, self.centroids, metric="euclidean")
            self.clusters = np.argmin(diff, axis=1)

            cluster_indices = []
            for j in range(self.clusters_number):
                cluster_indices.append(np.argwhere(self.clusters == j))

            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centers)


def read_image(file_path):
    image = cv.imread(file_path)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def preprocess_image(image):
    X = image.reshape((-1, 3))
    return np.float32(X)


def perform_segmentation(X, image, k):
    km = MyKMeans(clusters_number=k)
    km.fit(X)
    centers = km.centroids
    clusters = km.clusters
    segmented_image = centers[clusters]

    return segmented_image.reshape(image.shape)


def display_segmentation(image, segmented_image, k):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(segmented_image.astype(np.uint8))
    plt.title('Segmented Image when k = %i' % k), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    image = read_image('umbrellas.jpg')
    X = preprocess_image(image)
    while True:
        k = input('k = ')
        if k == '':
            break
        else:
           k = int(k)
           segmented_image = perform_segmentation(X, image, k)
           display_segmentation(image, segmented_image, k)
