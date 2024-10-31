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
        nearest_homogeneous, nearest_heterogeneous = None, None
        min_homogeneous_dist, min_heterogeneous_dist = np.inf, np.inf

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

import numpy as np
from collections import Counter

class EENTh:
    def __init__(self, k=3, threshold=None):
        self.k = k
        self.threshold = threshold

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_neighbors(self, X, y, sample):
        distances = [self._euclidean_distance(sample, X[i]) for i in range(len(X))]
        neighbors_idx = np.argsort(distances)[1:self.k+1]
        return [(y[i], distances[i]) for i in neighbors_idx]

    def _predict_proba(self, neighbors, y):
        classes = np.unique(y)
        counts = Counter([label for label, _ in neighbors])
        probs = np.array([counts[cls] / self.k for cls in classes])
        return probs

    def fit(self, X, y):
        S = np.copy(X)
        labels = np.copy(y)
        to_remove = []

        # Leave-one-out strategy
        for i in range(len(X)):
            # Exclude the current sample for leave-one-out
            X_without_i = np.delete(X, i, axis=0)
            y_without_i = np.delete(y, i, axis=0)

            # Find k nearest neighbors manually
            neighbors = self._get_neighbors(X_without_i, y_without_i, X[i])

            # Predict class probabilities for the current sample
            probs = self._predict_proba(neighbors, y_without_i)
            y_pred = np.argmax(probs)

            # Apply Wilson's Editing with threshold
            if y_pred != y[i] or (self.threshold is not None and np.max(probs) <= self.threshold):
                to_remove.append(i)

        # Remove misclassified or uncertain samples
        S = np.delete(S, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

        return S, labels

    
class DROP:
    def __init__(self, drop_type='drop1', k=3):
        self.k = k
        self.drop_type = drop_type

    def _classify_without(self, idx, exclude_idx):
        filtered_indices = [i for i in self.neighbors[idx] if i != exclude_idx][:self.k]
        return np.argmax(np.bincount(self.y[filtered_indices]))
    
    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_neighbors(self, X, sample):
        distances = [self._euclidean_distance(sample, X[i]) for i in range(len(X))]
        neighbors_idx = np.argsort(distances)[1:self.k+2]
        return neighbors_idx

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(int)

        # Initialize S with all indices
        S = list(range(len(X)))

        self.neighbors = {i: list(self._get_neighbors(X, X[i])) for i in S}
        
        # Noise filtering pass for DROP3
        if self.drop_type == 'drop3':
            S = self.noise_filtering_pass(S)

        # Create enemy distance list if using DROP2 or DROP3
        if self.drop_type in ['drop2', 'drop3']:
            distances_to_enemies = {
                i: np.min([np.linalg.norm(X[i] - X[j]) for j in S if y[i] != y[j]])
                for i in S
            }
            S = sorted(S, key=lambda x: distances_to_enemies[x], reverse=True)

        # Initialize associates for each instance in S
        associates = {i: set() for i in S}
        for p, ns in self.neighbors.items():
            for n in ns:
                if n in associates:
                    associates[n].add(p)

        # Evaluate each instance in the order of S
        for p in S[:]:  # Iterate over a copy of S to avoid modification during iteration

            with_correct = sum(1 for a in associates[p] if self._classify_without(a, None) == self.y[a])
            without_correct = sum(1 for a in associates[p] if self._classify_without(a, p) == self.y[a])

            # Remove P if classification accuracy does not degrade
            if without_correct >= with_correct:
                S.remove(p)
                
                # Update neighbors and associates after removing p
                for a in list(associates[p]):

                    self.neighbors[a].remove(p)
                    # Get the original index of 'a'
                    original_a_index = a
                    
                    # Get the subset of indices for S
                    S_indices = np.array(S)  # Get the indices of the current subset
                    S_data = self.X[S]  # Subset data using S indices
                    distances = np.linalg.norm(S_data - self.X[original_a_index], axis=1)  # Calculate distance to all instances in X[S]
                    sorted_indices = np.argsort(distances)  # Sort distances to get indices of neighbors

                    new_neighbor = [S_indices[idx] for idx in sorted_indices if S_indices[idx] != original_a_index and S_indices[idx] != p and S_indices[idx] not in self.neighbors[a]]
                    if new_neighbor:
                        self.neighbors[a].append(new_neighbor[0])
                        associates[new_neighbor[0]].add(a)
                
                if self.drop_type == 'drop1':
                    for neighbor in self.neighbors[p]:
                        if p in associates[neighbor]:
                            associates[neighbor].remove(p)
                    
                    associates[p].clear()

        X_prototypes = self.X[S]
        y_prototypes = self.y[S]
        return X_prototypes, y_prototypes

    def noise_filtering_pass(self, S):
        for p in S[:]:
            if self._classify_without(p, None) != self.y[p]:
                S.remove(p)
        return S