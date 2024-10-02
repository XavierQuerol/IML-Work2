from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression


def drop_columns(df, column_names):
    df = df.remove(column_names = column_names)
    return df

def drop_rows(df, column_names):
    df = df.dropna(subset=[column_names])
    return df

def min_max_scaler(df_train, df_test, column_names):
    ## berni
    return df_train, df_test

def one_hot_encoding(df_train, df_test):
    # select categorical features (excluding binary)
    categorical_features = df_train.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

    # one hot encoding
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(df_train[categorical_features])

    num_train = ohe.transform(df_train[categorical_features]).toarray()
    num_test = ohe.transform(df_test[categorical_features]).toarray()

    # add names to new features
    new_cols = [f'{col}_{cat}' for col in categorical_features for cat in
                ohe.categories_[categorical_features.index(col)]]
    df_train_encoded = pd.DataFrame(num_train, columns=new_cols)
    df_test_encoded = pd.DataFrame(num_test, columns=new_cols)

    # eliminate old features
    df_train = df_train.drop(categorical_features, axis=1)
    df_test = df_test.drop(categorical_features, axis=1)
    # add new features
    df_train = pd.concat([df_train, df_train_encoded], axis=1)
    df_test = pd.concat([df_test, df_test_encoded], axis=1)
    return df_train, df_test

def binary_encoding(df):
    ## angelica
    pass

def fill_nans(df_train, df_test, columns_train, columns_predict):
    model = LinearRegression()

    x = df_train[columns_train]
    y = df_train[columns_predict]

    model.fit(x, y)

    df_train[columns_predict] = model.predict(x)
    df_test[columns_predict] = model.predict(df_test[columns_train])

    return df_train, df_test