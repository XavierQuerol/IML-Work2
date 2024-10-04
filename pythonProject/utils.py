from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def drop_columns(df, column_names):
    df = df.remove(column_names = column_names)
    return df

def drop_rows(df, column_names):
    df = df.dropna(subset=[column_names])
    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df_train, df_test, numerical_cols):

    #Drop NaNs
    df_train = df_train.dropna(subset=numerical_cols)
    df_test = df_test.dropna(subset=numerical_cols)

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
        label_encoder.fit(df_train[feature])

        # Store the encoder in the dictionary
        label_encoders[feature] = label_encoder

        # Transform the training and test data
        df_train[feature] = label_encoder.transform(df_train[feature])
        df_test[feature] = label_encoder.transform(df_test[feature])

    return df_train, df_test


def fill_nans(df_train, df_test, columns_predict):

    model = LinearRegression()

    columns_train = [col for col in df_train.columns if col not in columns_predict]

    x = pd.concat((df_train[columns_train], df_test[columns_train]))
    y = pd.concat((df_train[columns_predict], df_test[columns_predict]))

    model.fit(x, y)

    df_train.loc[:, columns_predict] = model.predict(df_train[columns_train])
    df_test.loc[:, columns_predict] = model.predict(df_test[columns_train])

    return df_train, df_test
