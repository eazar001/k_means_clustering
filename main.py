import matplotlib
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
import glob
import os


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=400, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class KMeans:
    def __init__(self, k, rtol=1e-3):
        """
        :param k: Number of centroids
        :param rtol: Epsilon
        """
        self.k = k
        self.centroids = None
        self.snapshots = []  # buffer for progress plots
        self.rtol = rtol

    def initialize_centroids(self, X):
        """
        Randomly select k **distinct** samples from the dataset in X as centroids
        @param X: np.ndarray of dimension (num_samples, num_features)
        @return: centroids array of shape (k, num_features)
        """
        # Workspace I.1

        used = set()
        (n, m), centroids = X.shape, []

        while len(used) < self.k:
            s = X[np.random.randint(0, n)]

            if tuple(s) not in used:
                used.add(tuple(s))
                centroids.append(list(s))

        return np.vstack(centroids)

    def compute_distances(self, X):
        """
        Compute a distance matrix of size (num_samples, k) where each cell (i, j) represents the distance between
        i-th sample and j-th centroid. We shall use Euclidean distance here.
        :param X: np.ndarray of shape (num_samples, num_features)
        :return: distances_matrix : (np.ndarray) of the dimension (num_samples, k)
        """

        # Workspace I.2

        return euclidean_distances(X, self.centroids)

    @staticmethod
    def compute_assignments(distances_to_centroids):
        """
        Compute the assignment array of shape (num_samples,) where assignment[i] = j if and only if
        sample i belongs to the cluster of centroid j
        :param distances_to_centroids: The computed pairwise distances matrix of shape (num_samples, k)
        :return: assignments array of shape (num_samples,)
        """

        assignments = np.zeros((distances_to_centroids.shape[0],))

        # Workspace I.3

        n, k = distances_to_centroids.shape
        for i in range(n):
            ci, min_distance = 0, distances_to_centroids[i, 0]

            for j in range(k):
                if distances_to_centroids[i, j] < min_distance:
                    min_distance, ci = distances_to_centroids[i, j], j

            assignments[i] = ci

        return assignments.astype(int)

    def compute_centroids(self, X, assignments):
        """
        Given the assignments array for the samples, compute the new centroids
        :param X: data matrix of shape (num_samples, num_features)
        :param assignments: array of shape (num_samples,) where assignment[i] is the current cluster of sample i
        :return: The new centroids array of shape (k, num_features)
        """
        # Workspace I.4
        centroids = np.zeros((self.k, X.shape[1]))
        centroid_counts = np.zeros(self.k)

        n, m = X.shape

        for i in range(n):
            j = assignments[i]
            centroid_counts[j] += 1
            centroids[j] += X[i]

        for j in range(self.k):
            centroids[j] *= 1 / centroid_counts[j]

        return centroids

    def compute_objective(self, X, assignments):
        return np.sum(np.linalg.norm(X - self.centroids[assignments], axis=1) ** 2)

    def fit(self, X):
        """
        Implement the K-means algorithm here as described above. Loop until the improvement ratio of the objective
        is lower than rtol. At the end of each iteration, save the k-means objective and return the objective values
        at the end

        @param X:
        @return:
        """
        self.centroids = self.initialize_centroids(X)
        objective, new_objective = np.inf, np.inf
        history = []

        # Workspace I.5

        while objective == np.inf or abs(new_objective - objective) / abs(objective) >= self.rtol:
            objective = new_objective
            assignments = KMeans.compute_assignments(self.compute_distances(X))
            self.centroids = self.compute_centroids(X, assignments)
            new_objective = self.compute_objective(X, assignments)
            history.append(new_objective)
            self.snapshots.append(assignments)

        # for i, snapshot in enumerate(self.snapshots, start=1):
        #     plt.title(f'iteration_{i}, objective = {history[i - 1]}')
        #     plt.scatter(X[:, 0], X[:, 1], c=snapshot)
        #     plt.show()

        return history

    def predict(self, X):
        # Workspace I.6
        distances = euclidean_distances(X, self.centroids)
        n, _ = X.shape

        return np.array([np.argmin(distances[i]) for i in range(n)]).astype(int)


def evaluate_clustering(trained_model, X, labels):
    """
    Compute the ratio of correct matches between clusters from the trained model and the true labels
    :param trained_model: Unsupervised learning model that predicts clusters
    :param X: samples array, shape (num_samples, num_features)
    :param labels: true labels array, shape (num_samples,
    :return:
    """
    # We can assume that the number of clusters and the number of class labels are the same
    confusion_matrix = np.zeros((5, 5))
    boolean_matrix_X = np.zeros((5, 5))
    clusters = trained_model.predict(X)

    # Workspace I.7

    n, = labels.shape

    cluster_sets = dict([(x, set()) for x in range(5)])
    class_sets = dict([(x, set()) for x in range(5)])

    for i in range(n):
        cluster_sets[clusters[i]].add(i)
        class_sets[labels[i]].add(i)

    for i in range(5):
        for j in range(5):
            confusion_matrix[i, j] = len(cluster_sets[j] & class_sets[i])

    label_mappping, cluster_mapping = linear_sum_assignment(confusion_matrix, maximize=True)

    for i in range(5):
        boolean_matrix_X[label_mappping[i], cluster_mapping[i]] = 1

    return (confusion_matrix * boolean_matrix_X).sum() / confusion_matrix.sum()


def main():
    for f in glob.glob('iteration*'):
        os.remove(f)

    multi_blobs = DataBlobs(centers=5, std=1.5)

    # plt.title("multi_blobs")
    # plt.scatter(multi_blobs.X[:, 0], multi_blobs.X[:, 1], c=multi_blobs.labels)
    # plt.show()

    k_means = KMeans(5)

    accuracies = []

    for _ in range(20):
        k_means.fit(multi_blobs.X)
        accuracies.append(evaluate_clustering(k_means, multi_blobs.X, multi_blobs.labels))

    for accuracy in accuracies:
        print(accuracy)


if __name__ == '__main__':
    main()
