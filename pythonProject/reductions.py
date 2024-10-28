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
    

class DROP:
    def __init__(self, drop_type='drop1', k=3):
        self.k = k
        self.drop_type = drop_type

    def find_neighbors(self, instance_idx, exclude_idx=None):
        distances, indices = self.nbrs.kneighbors([self.X[instance_idx]])
        neighbors = indices[0][1:]  # Excluir la instancia misma
        if exclude_idx is not None:
            neighbors = [i for i in neighbors if i != exclude_idx]
        return neighbors

    def classify_without(self, idx, exclude_idx):
        distances, indices = self.nbrs.kneighbors([self.X[idx]])
        filtered_indices = [i for i in indices[0] if i != exclude_idx][:self.k]
        return np.argmax(np.bincount(self.y[filtered_indices]))

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(int)
        self.original_indices = list(range(len(X)))

        self.nbrs = KNeighborsClassifier(n_neighbors=self.k + 1).fit(X,y)
        
        # Inicializar S con todos los índices
        S = list(range(len(self.X)))
        
        # Filtro de ruido si es DROP3
        if self.drop_type == 'drop3':
            S = self.noise_filtering_pass(S)

        # Crear una lista de distancias a los enemigos más cercanos
        if self.drop_type in ['drop2', 'drop3']:
            distances_to_enemies = {
                i: np.min([np.linalg.norm(self.X[i] - self.X[j]) for j in S if self.y[i] != self.y[j]])
                for i in S
            }
            S = sorted(S, key=lambda x: distances_to_enemies[x])

        # Diccionario para los asociados
        associates = {i: set() for i in S}
        
        # Ajustar el clasificador para el conjunto S inicial
        self.nbrs = KNeighborsClassifier(n_neighbors=self.k + 1).fit(self.X[S], self.y[S])

        # Crear la lista de asociados inicial
        for p in S:
            neighbors = self.find_neighbors(p)
            for n in neighbors:
                if n in associates:
                    associates[n].add(p)
                else:
                    associates[n] = {p}

        # Evaluar cada instancia en el orden de S
        for p in S[:]:  # Copia de S para no modificarlo durante la iteración
            with_correct = sum(1 for a in associates[p] if self.classify_without(a, None) == self.y[a])
            without_correct = sum(1 for a in self.original_indices if a in associates[p] and self.classify_without(a, p) == self.y[a])

            # Remover P si la precisión de clasificación no se degrada
            if without_correct >= with_correct:
                S.remove(p)
                
                # Ajustar el clasificador con el nuevo S
                self.nbrs = KNeighborsClassifier(n_neighbors=self.k + 1).fit(self.X[S], self.y[S])

                # Actualizar la lista de vecinos y asociados
                for a in list(associates[p]):
                    if p in associates[a]:
                        associates[a].remove(p)
                        new_neighbors = self.find_neighbors(a, exclude_idx=p)
                        for new_neighbor in new_neighbors:
                            associates[new_neighbor].add(a)
                associates[p].clear()

        # Ajustar el clasificador final con el conjunto reducido S
        self.nbrs = KNeighborsClassifier(n_neighbors=self.k + 1).fit(self.X[S], self.y[S])
        X_prototypes = self.X[S]
        y_prototypes = self.y[S]
        return X_prototypes, y_prototypes

    def noise_filtering_pass(self, S):
        for p in S[:]:
            if self.classify_without(p, None) != self.y[p]:
                S.remove(p)
        return S
    
    def drop3(self):
        """Perform DROP3 algorithm for instance reduction and return prototypes."""
        S = list(range(len(self.X)))  # Initialize S to include all instances

        # Step 1: Noise filtering pass
        S = self.noise_filtering_pass(S)

        # Step 2: Create a distance list to the nearest enemy for sorting
        distances_to_enemies = {
            i: np.min([np.linalg.norm(self.X[i] - self.X[j]) for j in S if self.y[i] != self.y[j]])
            for i in S
        }
        
        # Sort instances in S by the distance to their nearest enemy
        sorted_S = sorted(S, key=lambda x: distances_to_enemies[x])

        # Step 3: Evaluate each instance in the sorted order
        associates = {i: set() for i in S}  # Dictionary to track associates of each instance
        
        # Build associates list for each instance in S
        

        for p in S:
            neighbors = self.find_neighbors(p)
            for n in neighbors:
                if n in associates:  # Only add if n already exists in associates
                    associates[n].add(p)  # Add p to each of its neighbors’ lists of associates
                else:
                    associates[n] = {p}  # Initialize a new set for n if not present

        for p in sorted_S:
            with_correct = sum(1 for a in associates[p] if self.classify_without(a, None) == self.y[a])
            # Check associates in the original set T instead of S
            without_correct = sum(1 for a in self.original_indices if a in associates[p] and self.classify_without(a, p) == self.y[a])

            # Remove P if the classification accuracy does not degrade
            if without_correct >= with_correct:
                S.remove(p)  # Remove p from S

                # Update the associates lists and neighbors for each associate of P
                for a in list(associates[p]):  # Use a copy of associates[p] to avoid modifying during iteration
                    if p in associates[a]:
                        associates[a].remove(p)  # Remove P from A’s list of nearest neighbors
                        new_neighbors = self.find_neighbors(a, exclude_idx=p)  # Find new nearest neighbors
                        for new_neighbor in new_neighbors:
                            associates[new_neighbor].add(a)  # Add A to its new neighbor’s list of associates

                # Clean up associates of P itself
                associates[p].clear()

        # Refit NearestNeighbors with the reduced set S once at the end
        self.nbrs = KNeighborsClassifier(n_neighbors=self.k + 1).fit(self.X[S], self.y[S])

        # Return X and y subsets based on reduced set S
        X_prototypes = self.X[S]
        y_prototypes = self.y[S]
        return X_prototypes, y_prototypes