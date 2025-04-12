import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

class SoftCluster:
    def __init__(self, n_clusters=None, eigengap_threshold=0.1):
        """
        Initialize the SoftCluster class.

        Parameters:
        - n_clusters: Number of clusters. If None, it will be determined using the eigengap heuristic.
        - eigengap_threshold: Threshold for determining the optimal number of clusters using the eigengap.
        """
        self.n_clusters = n_clusters
        self.eigengap_threshold = eigengap_threshold

    def _compute_eigengap(self, eigenvalues):
        """
        Compute the eigengap to determine the optimal number of clusters.

        Parameters:
        - eigenvalues: Sorted eigenvalues of the Laplacian matrix.

        Returns:
        - Optimal number of clusters based on the eigengap heuristic.
        """
        gaps = np.diff(eigenvalues)
        optimal_clusters = np.argmax(gaps > self.eigengap_threshold) + 1
        return optimal_clusters

    def fit(self, X):
        """
        Fit the soft clustering model to the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - cluster_probabilities: Soft cluster assignments, shape (n_samples, n_clusters).
        """
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(X)

        # Compute eigenvalues and eigenvectors of the Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
        eigenvalues = np.sort(eigenvalues)

        # Determine the number of clusters using eigengap heuristic if not provided
        if self.n_clusters is None:
            self.n_clusters = self._compute_eigengap(eigenvalues)

        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', assign_labels='discretize')
        labels = spectral.fit_predict(similarity_matrix)

        # Compute soft cluster probabilities
        cluster_probabilities = np.zeros((X.shape[0], self.n_clusters))
        for i, label in enumerate(labels):
            cluster_probabilities[i, label] = 1.0

        return cluster_probabilities