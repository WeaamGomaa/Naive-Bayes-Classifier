import numpy as np

class PCA:
    def __init__(self, components_num):
        self.components_num = components_num
        self.components = None
        self.mean = None

    def fit(self, X):
        #subtract the mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        #calculate covariance, functions needs samples as columns
        cov = np.cov(X.T)

        #calculate eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        #sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.components_num]

    def transform(self, X):
        #projects data
        X = X - self.mean
        return np.dot(X, self.components.T)








