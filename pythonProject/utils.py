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

def one_hot_encoding(df):
    ## aina
    pass

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