import numpy as np

class PCA:
    def __init__(self, components_num):
        self.components_num = components_num
        self.components = None
        self.mean = None

    def fit(self, X):
        # Subtract the mean (centring data)
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculate covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # Calculate eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # Sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.components_num]

    def transform(self, X):
        # Projects data
        X = X - self.mean
        return np.dot(X, self.components.T)








