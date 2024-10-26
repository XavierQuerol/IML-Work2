import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class CNN_GCNN:
    def __init__(self, rho=0):
        self.rho = rho  # GCNN's parameter for controlling the absorption criterion
        self.prototypes = None

    def fit(self, X_train, y_train):
        """Condensed Nearest Neighbor algorithm (CNN) with GCNN extension."""

        # CNN Step 1: Initiate with one random prototype for each class
        labels = np.unique(y_train)
        prototypes_X = []
        prototypes_y = []

        for label in labels:
            # Randomly pick one sample per class to be the initial prototype
            idx = np.random.choice(np.where(y_train == label)[0])
            prototypes_X.append(X_train[idx])
            prototypes_y.append(y_train[idx])

        # Keep track of prototype set Un
        prototypes_X = np.array(prototypes_X, dtype=object)
        prototypes_y = np.array(prototypes_y, dtype=object)

        # Compute delta_n (GCNN)
        delta_n = self._compute_delta_n(X_train, y_train)

        absorbed = np.zeros(X_train.shape[0], dtype=bool)  # Track absorbed samples

        while not np.all(absorbed):
            for i, (x_i, y_i) in enumerate(zip(X_train, y_train)):
                if not absorbed[i]:  # If sample i is not absorbed
                    nearest_homogeneous, nearest_heterogeneous = self._find_nearest_prototypes(x_i, y_i, (
                    prototypes_X, prototypes_y))

                    # CNN absorption criterion (rho=0 corresponds to CNN)
                    if self._should_absorb(nearest_homogeneous, nearest_heterogeneous, x_i, delta_n):
                        absorbed[i] = True
                    else:
                        # If not absorbed, add the sample as a new prototype
                        prototypes_X = np.append(prototypes_X, np.array([x_i]), axis=0)
                        prototypes_y = np.append(prototypes_y, np.array([y_i]), axis=0)

        # Store the prototypes for prediction
        return prototypes_X, prototypes_y

    def _compute_delta_n(self, X, y):
        """Compute delta_n, the minimum distance between samples of different labels."""
        min_distance = np.inf
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if y[i] != y[j]:
                    dist = np.linalg.norm(X[i] - X[j])
                    if dist < min_distance:
                        min_distance = dist
        return min_distance

    def _find_nearest_prototypes(self, x, y, prototypes):
        """Find the nearest homogeneous and heterogeneous prototypes."""
        nearest_homogeneous = None
        nearest_heterogeneous = None
        min_homogeneous_dist = np.inf
        min_heterogeneous_dist = np.inf

        prototypes_X = prototypes[0]
        prototypes_y = prototypes[1]

        for i in range(prototypes_X.shape[0]):
            proto_x, proto_y = prototypes_X[i], prototypes_y[i]
            dist = np.linalg.norm(x - proto_x)
            if proto_y == y and dist < min_homogeneous_dist:
                nearest_homogeneous = proto_x
                min_homogeneous_dist = dist
            elif proto_y != y and dist < min_heterogeneous_dist:
                nearest_heterogeneous = proto_x
                min_heterogeneous_dist = dist

        return nearest_homogeneous, nearest_heterogeneous

    def _should_absorb(self, nearest_homogeneous, nearest_heterogeneous, x, delta_n):
        """GCNN absorption criterion (rho=0 corresponds to CNN)."""
        d_homogeneous = np.linalg.norm(x - nearest_homogeneous)
        d_heterogeneous = np.linalg.norm(x - nearest_heterogeneous)
        return d_homogeneous < d_heterogeneous - self.rho * delta_n

class EENTh:
    def __init__(self, k=3, threshold=None):
        self.k = k
        self.threshold = threshold

    def fit(self, X, y):
        knn = KNeighborsClassifier(n_neighbors=self.k)
        S = np.copy(X)
        labels = np.copy(y)

        to_remove = []

        # Leave-one-out strategy
        for i in range(len(X)):
            # Exclude the current sample for leave-one-out
            X_without_i = np.delete(X, i, axis=0)
            y_without_i = np.delete(y, i, axis=0)

            # Train k-NN without the current sample
            knn.fit(X_without_i, y_without_i)

            # Predict class probabilities for the current sample
            probs = knn.predict_proba([X[i]])[0]
            y_pred = np.argmax(probs)


            # Apply WilsonTh if threshold is given, otherwise Wilson's Editing
            if y_pred != y[i] or (self.threshold is not None and np.max(probs) <= self.threshold):
                to_remove.append(i)

        # Remove misclassified or uncertain samples
        S = np.delete(S, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

        return S, labels