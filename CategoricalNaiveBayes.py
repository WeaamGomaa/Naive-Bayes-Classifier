import numpy as np

class CategoricalNaiveBayes:
    def __init__(self, laplace_smoothing=1):
        # Laplace smoothing prevents zero probabilities
        # when a value wasn't seen during training
        self.laplace_smoothing = laplace_smoothing
        self._classes = None
        self._priors = None
        self._likelihoods = None

    def fit(self, X, Y):
        # Convert to numpy arrays to be safe
        X = np.array(X)
        Y = np.array(Y)

        samples_num, features_num = X.shape
        self._classes = np.unique(Y)       # [0, 1]
        classes_num = len(self._classes)   # 2

        # Step 1: Calculate prior probabilities P(class)
        # Prior = how often each class appears in training data
        self._priors = np.zeros(classes_num)
        for idx, c in enumerate(self._classes):
            # Count how many samples belong to class c
            self._priors[idx] = np.sum(Y == c) / samples_num
            # Example: 3000 edible / 4874 total = 0.615

        # Step 2: Calculate likelihoods P(feature=value | class)
        # For each feature, for each class, for each possible value
        # we count how often that value appears
        self._likelihoods = []

        for feature_idx in range(features_num):
            # Get all unique values this feature can take
            feature_values = np.unique(X[:, feature_idx])
            feature_likelihoods = {}

            for idx, c in enumerate(self._classes):
                # Get all rows belonging to class c
                X_c = X[Y == c]
                class_count = X_c.shape[0]  # total samples in this class
                feature_likelihoods[c] = {}

                for value in feature_values:
                    # Count how many times this value appears for this class
                    value_count = np.sum(X_c[:, feature_idx] == value)

                    # Apply Laplace smoothing:
                    # Without smoothing: if value never appeared → probability = 0
                    # With smoothing: add 1 to numerator, add num_values to denominator
                    num_unique_values = len(feature_values)
                    probability = (value_count + self.laplace_smoothing) / \
                                  (class_count + self.laplace_smoothing * num_unique_values)

                    feature_likelihoods[c][value] = probability

            self._likelihoods.append(feature_likelihoods)

    def predict(self, X):
        X = np.array(X)
        # Predict class for every row
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # Calculate posterior for each class and pick the highest
        posteriors = []

        for idx, c in enumerate(self._classes):
            # Start with log(prior) — use log to avoid tiny float underflow
            posterior = np.log(self._priors[idx])

            # Multiply by likelihood of each feature value
            # In log space: multiplication becomes addition
            for feature_idx, value in enumerate(x):
                likelihoods_for_feature = self._likelihoods[feature_idx][c]

                if value in likelihoods_for_feature:
                    # Value was seen during training
                    posterior += np.log(likelihoods_for_feature[value])
                else:
                    # Value was never seen during training
                    # Use a small smoothed probability instead of 0
                    posterior += np.log(self.laplace_smoothing /
                                       (len(likelihoods_for_feature) + self.laplace_smoothing))

            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]