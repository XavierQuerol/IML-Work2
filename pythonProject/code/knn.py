import os

import numpy as np
import pandas as pd
import time

from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn_relief import Relief

from utils import computeMetrics

def callKNNs(X_train, X_test, y_train, y_test, ds_name, fold):

    columns=['K','Distance', 'Voting scheme', 'Weight scheme', 'Solving Time', 'Accuracy']
    classes = y_train.unique()

    for i in classes:
        columns.append(f'Precision_Class_{i}')
        columns.append(f'Recall_Class_{i}')
        columns.append(f'F1_Class_{i}')

    results = pd.DataFrame(columns=columns)
    distance_functions = ['minkowski1','minkowski2','HEOM']
    voting_schemes = ['Majority_class','Inverse_Distance_Weights', 'Sheppards_Work']
    weight_schemes = ['Mutual_classifier','Relief','ANOVA']
    ks = [1,3,5,7]

    for dist_func in distance_functions:
        for voting_scheme in voting_schemes:
            for weight_scheme in weight_schemes:
                for k in ks:
                    print(f" - Using distance {dist_func} - voting {voting_scheme} - weighting {weight_scheme} - K {k}")
                    start = time.time()
                    knn = KNN(dist_func, voting_scheme, weight_scheme,k)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    solving_time = time.time() - start
                    accuracy, precision, recall, f1 = computeMetrics(y_test, y_pred)
                    res = {'K': k, 'Distance': dist_func, 'Voting scheme': voting_scheme, 'Weight scheme': weight_scheme, 'Accuracy': accuracy,'Solving Time': solving_time}
                    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                        res[f'Precision_Class_{i}'] = p
                        res[f'Recall_Class_{i}'] = r
                        res[f'F1_Class_{i}'] = f

                    new_row = pd.DataFrame([res])

                    results = pd.concat([results.astype(new_row.dtypes), new_row.astype(results.dtypes)], ignore_index=True)

                    print(f"This combination took {solving_time} seconds")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_knn")
    results.to_csv(os.path.join(base_dir, f'results_{ds_name}_{fold}.csv'), index=False)


def callKNN(X_train, X_test, y_train, y_test, dist_func, voting_scheme, weight_scheme, k):

    columns=['K','Distance', 'Voting scheme', 'Weight scheme', 'Solving Time', 'Accuracy', 'Num samples']
    classes = y_train.unique()

    for i in classes:
        columns.append(f'Precision_Class_{i}')
        columns.append(f'Recall_Class_{i}')
        columns.append(f'F1_Class_{i}')
    results = pd.DataFrame(columns=columns)

    start = time.time()
    knn = KNN(dist_func, voting_scheme, weight_scheme,k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    solving_time = time.time() - start
    accuracy, precision, recall, f1 = computeMetrics(y_test, y_pred)
    res = {'K': k, 'Distance': dist_func, 'Voting scheme': voting_scheme, 'Weight scheme': weight_scheme, 'Accuracy': accuracy,'Solving Time': solving_time, 'Num samples': len(X_train)}
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        res[f'Precision_Class_{i}'] = p
        res[f'Recall_Class_{i}'] = r
        res[f'F1_Class_{i}'] = f

    new_row = pd.DataFrame([res])

    results = pd.concat([results.astype(new_row.dtypes), new_row.astype(results.dtypes)], ignore_index=True)

    return results


# KNN

class KNN:
    def __init__(self, distance_function, voting_scheme, weight_scheme, k=3):
        self.k = k
        self.X_train = None
        self.X_train_weighted = None
        self.y_train = None

        self.distance_functions = {'minkowski1': Distances.minkowski1, 'minkowski2': Distances.minkowski2, 'HEOM': Distances.HEOM}
        self.voting_schemes = {'Majority_class': Voting_schemes.majority_class, 'Inverse_Distance_Weights': Voting_schemes.inverse_distance_weight, 'Sheppards_Work': Voting_schemes.sheppards_work}
        self.weight_schemes = {'Mutual_classifier': Weighting.update_weights_mutual_classifier, 'Relief': Weighting.update_weights_relief, 'ANOVA': Weighting.update_weights_anova}

        self.distance_function = self.distance_functions[distance_function]
        self.voting_scheme = self.voting_schemes[voting_scheme]
        self.weight_scheme = self.weight_schemes[weight_scheme]

        self.class_weights = None

    def compute_class_weights(self, train_labels):
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / count for cls, count in zip(unique_classes, class_counts)}
        return class_weights

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.class_weights = self.compute_class_weights(y_train)

    def predict(self, X_test):

        # Use numpy
        self.X_train_weighted, X_test = self.weight_scheme(self.X_train, self.y_train, X_test)
        predictions = [self._predict(x) for x in X_test.values]
        return predictions

    def _predict(self, x):

        # Calculate distances between x and all examples in the training set
        # Use numpy
        distances = np.array([self.distance_function(x, x_train) for x_train in self.X_train_weighted.values])

        # Get the indices of the k-nearest neighbors
        sorted_indices  = np.argsort(distances)

        k_indices = sorted_indices[:self.k]
        k_nearest_distances = distances[sorted_indices[:self.k]]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Choose between voting schemes
        predicted_class = self.voting_scheme(k_nearest_distances, k_nearest_labels)
        return predicted_class

# Distance Metrics:
# Distance Metrics:
class Distances:
    @staticmethod
    def minkowski2(a, b):
        return np.sqrt(np.sum(np.power(np.abs(a - b),2)))


    @staticmethod
    def minkowski1(a, b):
        return np.sum(np.abs(a - b))


    @staticmethod
    def minkowski(a, b, p):
        return np.power(np.sum(np.power(np.abs(a - b),p)), (1 / p))

    @staticmethod
    def HEOM(a, b):
        """Heterogeneous Euclidean-Overlap Metric (HEOM) distance, because it takes into account if
        features are numerical or categorical.
        """
        a = np.asarray(a)
        b = np.asarray(b)

        num_features = len(a)
        distance = 0

        for i in range(num_features):
            if all(x in (0, 1) for x in [a[i], b[i]]):  # check if both are 0 or 1 tp see if they are categorical
                # Overlap metric for categorical features
                distance += (a[i] != b[i])  # 1 if different, 0 if same
            else:
                # Euclidean distance for numerical features
                distance += (a[i] - b[i]) ** 2

        return np.sqrt(distance)

    @staticmethod
    def mahalanobis(a, b, cov_inv):
        """Mahalanobis distance between points a and b with a given covariance matrix."""
        a = np.asarray(a)
        b = np.asarray(b)
        delta = a - b
        distance = np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))
        return distance



# Voting schemes

# Common helper to handle tie-breaking
def handle_tie(classes, metric):
    return classes[np.argmax(metric)]  # Breaking ties by the largest metric

class Voting_schemes:
    # distances: list of distances to the k nearest neighbours
    # classes: list of classes of the k nearest neighbours
    # class_weights: dictionary of class weights (optional)
    @staticmethod
    def majority_class(distances, classes, class_weights=None):
        unique_classes, count = np.unique(classes, return_counts=True)

        # Apply class weights if provided
        if class_weights is not None:
            count = np.array([count[i] * class_weights.get(cls, 1) for i, cls in enumerate(unique_classes)])

        max_count = np.max(count)
        max_class = unique_classes[count == max_count]

        if len(max_class) == 1:
            return max_class[0]

        # Tie breaking by average distance
        avg_distances = np.array([np.mean(distances[classes == cls]) for cls in max_class])
        max_class = handle_tie(max_class, -avg_distances)

        return max_class


    # distances: list of distances to the k nearest neighbours
    # classes: list of classes of the k nearest neighbours
    # class_weights: dictionary of class weights (optional)
    @staticmethod
    def inverse_distance_weight(distances, classes, class_weights=None):
        unique_classes = np.unique(classes)
        metric = np.zeros(len(unique_classes))

        for i, cls in enumerate(unique_classes):
            d = distances[classes == cls]

            if np.any(d == 0):
                return cls
            inverse_d = 1 / d
            metric[i] = np.sum(inverse_d)


        # Apply class weights if provided
        if class_weights is not None:
            metric = np.array([metric[i] * class_weights.get(cls, 1) for i, cls in enumerate(unique_classes)])

        max_count = np.max(metric)
        max_class = unique_classes[metric == max_count]

        if len(max_class) == 1:
            return max_class[0]

        # Tie breaking by the metric
        max_class = handle_tie(max_class, metric)
        return max_class


    # distances: list of distances to the k nearest neighbours
    # classes: list of classes of the k nearest neighbours
    # class_weights: dictionary of class weights (optional)
    @staticmethod
    def sheppards_work(distances, classes, class_weights=None):
        unique_classes = np.unique(classes)
        metric = np.zeros(len(unique_classes))

        for i, cls in enumerate(unique_classes):
            d = distances[classes == cls]
            metric[i] = np.sum(np.exp(-d))

        # Apply class weights if provided
        if class_weights is not None:
            metric = np.array([metric[i] * class_weights.get(cls, 1) for i, cls in enumerate(unique_classes)])

        max_count = np.max(metric)
        max_class = unique_classes[metric == max_count]

        if len(max_class) == 1:
            return max_class[0]

        # Tie breaking by the metric
        max_class = handle_tie(max_class, metric)
        return max_class


## Weighting:
class Weighting:
    @staticmethod
    def update_weights_mutual_classifier(X_train, y_train, X_test=None):

        # Compute information gain
        mi = mutual_info_classif(X_train, y_train)

        # Scale the values of the columns by their information gain.
        X_train_weighted = X_train * mi
        X_test_weighted = X_test * mi

        return X_train_weighted, X_test_weighted

    # We have to use Relief and not Relieff because our problems have only 2 classes.
    # The core idea behind Relief algorithms is to estimate the quality of attributes on the basis of how well the attribute can distinguish between instances that are near to each other.
    # https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83
    @staticmethod
    def update_weights_relief(X_train, y_train, X_test=None):

        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
        X_test_np = X_test.to_numpy()

        relief = Relief()

        relief.fit(X_train_np, y_train_np)

        X_train_weighted = relief.transform(X_train_np)
        X_test_weighted = relief.transform(X_test_np)
        X_train_weighted = pd.DataFrame(X_train_weighted)
        X_test_weighted = pd.DataFrame(X_test_weighted)

        return X_train_weighted, X_test_weighted

    # Only works with numeric data
    # Ranks features by how much they distinguish between the target classes based on variance between groups
    @staticmethod
    def update_weights_anova(X_train, y_train, X_test=None):

        k = len(X_train.columns) // 2

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selected_features_mask = selector.get_support()

        X_train_selected = X_train.loc[:, selected_features_mask]
        X_test_selected = X_test.loc[:, selected_features_mask]

        return X_train_selected, X_test_selected