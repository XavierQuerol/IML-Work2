import time

from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn_relief import Relief
import pandas as pd
import numpy as np


def drop_columns(df, column_names):
    df = df.drop(columns = column_names)
    return df

def drop_rows(df, column_names):
    df = df.dropna(subset=column_names)
    df = df.reset_index(drop=True)
    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df_train, df_test, numerical_cols=slice(None)):

    scaler = MinMaxScaler()

    # Scaler Training with all the train and test information.
    scaler.fit(pd.concat((df_train[numerical_cols], df_test[numerical_cols])))

    # Scale train and test data separately
    df_train.loc[:, numerical_cols] = scaler.transform(df_train[numerical_cols])
    df_test.loc[:, numerical_cols] = scaler.transform(df_test[numerical_cols])

    return df_train, df_test

def one_hot_encoding(df_train, df_test):
  # select categorical features (excluding binary)
  categorical_features = df_train.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

  # one hot encoding
  ohe = OneHotEncoder(handle_unknown='ignore')
  # Correctly pass a list of dataframes to pd.concat
  ohe.fit(pd.concat([df_train[categorical_features], df_test[categorical_features]]))

  num_train = ohe.transform(df_train[categorical_features]).toarray()
  num_test = ohe.transform(df_test[categorical_features]).toarray()

  # add names to new features
  new_cols = [f'{col}_{cat}' for col in categorical_features for cat in ohe.categories_[categorical_features.index(col)]]
  df_train_encoded = pd.DataFrame(num_train, columns=new_cols)
  df_test_encoded = pd.DataFrame(num_test, columns=new_cols)

  # eliminate old features
  df_train = df_train.drop(categorical_features, axis=1)
  df_test = df_test.drop(categorical_features, axis=1)
  # add new features
  df_train = pd.concat([df_train, df_train_encoded], axis=1)
  df_test = pd.concat([df_test, df_test_encoded], axis=1)
  return df_train, df_test

def binary_encoding(df_train, df_test):
    # select binary features
    binary_features = df_train.select_dtypes(include=['object']).nunique()[lambda x: x <= 2].index.tolist()

    label_encoders = {}

    for feature in binary_features:
        label_encoder = LabelEncoder()

        # Fit the encoder only to the training data
        label_encoder.fit(pd.concat([df_train[feature], df_test[feature]]))

        # Store the encoder in the dictionary
        label_encoders[feature] = label_encoder

        # Transform the training and test data
        df_train[feature] = label_encoder.transform(df_train[feature])
        df_test[feature] = label_encoder.transform(df_test[feature])

    return df_train, df_test


def fill_nans(df_train, df_test, columns_predict):

    model = LinearRegression()

    columns_train = [col for col in df_train.columns if col not in columns_predict]
    for col in columns_predict:
        df_train_model = df_train.dropna(subset=[col])
        df_train_nans = df_train[df_train[col].isna()]
        df_test_model = df_test.dropna(subset=[col])
        df_test_nans = df_test[df_train[col].isna()]
        x = pd.concat((df_train_model[columns_train], df_test_model[columns_train]))
        y = pd.concat((df_train_model[col], df_test_model[col]))

        model.fit(x, y)

        df_train.loc[df_train_nans.index, col] = model.predict(df_train_nans[columns_train])
        df_test.loc[df_test_nans.index, col] = model.predict(df_test_nans[columns_train])

    return df_train, df_test


## SESSION 2:

def computeMetrics():
    pass

def callKNNs(X_train, X_test, y_train, y_test):
    distance_functions = []
    voting_schemes = []
    weight_schemes = []
    results = pd.DataFrame()

    for dist_func in distance_functions:
        for voting_scheme in voting_schemes:
            for weight_scheme in weight_schemes:
                start = time.time()
                knn = KNN(dist_func, voting_scheme, weight_scheme,3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                solving_time = time.time() - start
                results.append({'Distance': dist_func, 'Voting scheme': voting_scheme, 'Weight scheme': weight_scheme, 'Metrics': computeMetrics(y_test,y_pred),'Solving Time': solving_time}) # Cambiar

# KNN

class KNN:
    def __init__(self, distance_function, voting_scheme, weight_scheme, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

        self.distance_functions = {'minkowski1': Distances.minkowski1, 'minkowski2': Distances.minkowski2, 'HEOM': Distances.HEOM}
        self.voting_schemes = {'Majority_class': Voting_schemes.majority_class, 'Inverse_Distance_Weights': Voting_schemes.inverse_distance_weight, 'Sheppards_Work': Voting_schemes.sheppards_work}
        self.weight_schemes = {'Mutual_classifier': Weighting.update_weights_mutual_classifier, 'Relief': Weighting.update_weights_relief, 'ANOVA': Weighting.update_weights_anova}

        self.distance_function = self.distance_functions[distance_function]
        self.voting_scheme = self.voting_schemes[voting_scheme]
        self.weight_scheme = self.weight_schemes[weight_scheme]

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):

        # Use numpy
        self.weight_scheme(self.X_train, self.y_train, X_test)
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):

        # Calculate distances between x and all examples in the training set
        # Use numpy
        distances = [self.distance_function(x, x_train) for x_train in self.X_train]

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
    def minkowski2(self, a, b):
        return minkowski(a, b, 2)


    def minkowski1(self, a, b):
        return minkowski(a, b, 1)


    def minkowski(self, a, b, r):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.sum(np.abs(a - b) ** r) ** (1 / r)

    def HEOM(self, a, b):
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

# Voting schemes

# Common helper to handle tie-breaking
def handle_tie(classes, metric):
    return classes[np.argmax(metric)]  # Breaking ties by the largest metric

class Voting_schemes:
    # distances: list of distances to the k nearest neighbours
    # classes: list of classes of the k nearest neighbours
    # class_weights: dictionary of class weights (optional)
    def majority_class(self, distances, classes, class_weights=None):
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
    def inverse_distance_weight(self, distances, classes, class_weights=None):
        unique_classes = np.unique(classes)
        metric = np.zeros(len(unique_classes))

        for i, cls in enumerate(unique_classes):
            d = distances[classes == cls]
            metric[i] = np.sum(1 / d)

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
    def sheppards_work(self, distances, classes, class_weights=None):
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
    def update_weights_mutual_classifier(self, X_train, y_train, X_test):

        # Compute information gain
        mi = mutual_info_classif(X_train, y_train)

        # Scale the values of the columns by their information gain.
        X_train_weighted = X_train * mi
        X_test_weighted = X_test * mi

        return X_train_weighted, X_test_weighted

    # We have to use Relief and not Relieff because our problems have only 2 classes.
    # The core idea behind Relief algorithms is to estimate the quality of attributes on the basis of how well the attribute can distinguish between instances that are near to each other.
    # https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83
    def update_weights_relief(self, X_train, y_train, X_test):

        relief = Relief()
        relief.fit(X_train, y_train)
        X_train_weighted = relief.transform(X_train)
        X_test_weighted = relief.transform(X_test)

        return X_train_weighted, X_test_weighted

    # Only works with numeric data
    # Ranks features by how much they distinguish between the target classes based on variance between groups
    def update_weights_anova(self, X_train, y_train, X_test, k=10):

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        selected_features_mask = selector.get_support()

        X_train_selected = X_train.loc[:, selected_features_mask]
        X_test_selected = X_test.loc[:, selected_features_mask]

        return X_train_selected, X_test_selected