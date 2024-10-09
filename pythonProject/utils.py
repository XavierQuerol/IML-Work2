from scipy.spatial.distance import minkowski
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
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
def min_max_scaler(df_train, df_test, numerical_cols):

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

# KNN

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, distance_function: callable(), voting_function: callable()):

        # Use numpy
        predictions = [self._predict(x, distance_function, voting_function) for x in X_test]
        return predictions

    def _predict(self, x, distance_function: callable(), voting_function: callable()):

        # Calculate distances between x and all examples in the training set
        # Use numpy
        distances = [distance_function(x, x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest neighbors
        sorted_indices  = np.argsort(distances)

        k_indices = sorted_indices[:self.k]
        k_nearest_distances = distances[sorted_indices[:self.k]]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Choose between voting schemes
        predicted_class = voting_function(k_nearest_distances, k_nearest_labels)
        return predicted_class

# Distance Metrics:
def minkowski2(a,b) :
    return minkowski(a,b,2)

def minkowski1(a,b):
    return minkowski(a,b,1)

def minkowski(a, b, r):
    pass

def metric2(a, b):
    pass

# Voting schemes

# distances: list of distances to the k nearest neighbours
# classes: list of classes of the k nearest neighbours
def majority_class(distances, classes):
    # Tie breaking
    pass

# distances: list of distances to the k nearest neighbours
# classes: list of classes of the k nearest neighbours
def inverse_distance_weight(distances, classes):
    # Tie breaking
    pass

# distances: list of distances to the k nearest neighbours
# classes: list of classes of the k nearest neighbours
def sheppards_work(distances, classes):
    # Tie breaking
    pass
