from scipy.io.arff import loadarff
import pandas as pd

from utils import drop_rows, drop_columns, min_max_scaler, one_hot_encoding, binary_encoding, fill_nans


for fold in range(10):
    df_sick_train, meta_train = loadarff(f'sick/sick.fold.00000{fold}.train.arff')
    df_sick_test, meta_test = loadarff(f'sick/sick.fold.00000{fold}.test.arff')

    df_sick_train = pd.DataFrame(df_sick_train)
    df_sick_test = pd.DataFrame(df_sick_test)
    df_sick_train = df_sick_train.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df_sick_test = df_sick_test.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    columns_with_nans = []
    for col in df_sick_train.columns:
        sum_nans = sum(df_sick_train[col].isna())
        percentage = sum_nans / len(df_sick_train) * 100
        print(f'{col}: {percentage}')
        if (percentage > 0.1) and (percentage < 99):
            columns_with_nans.append(col)

    df_sick_train = drop_rows(df_sick_train, ['age'])
    df_sick_test = drop_rows(df_sick_test, ['age'])

    df_sick_train = drop_columns(df_sick_train, ['TBG_measured', 'TBG'])
    df_sick_test = drop_columns(df_sick_test, ['TBG_measured', 'TBG'])

    df_sick_train, df_sick_test = min_max_scaler(df_sick_train, df_sick_test, ['age'])

    df_sick_train, df_sick_test = one_hot_encoding(df_sick_train, df_sick_test)

    df_sick_train, df_sick_test = binary_encoding(df_sick_train, df_sick_test)

    df_sick_train, df_sick_test = fill_nans(df_sick_train, df_sick_test, columns_with_nans)

    df_sick_train, df_sick_test = min_max_scaler(df_sick_train, df_sick_test, columns_with_nans)

    df_sick_train.to_csv(f'sick_results/sick.fold.00000{fold}.train.csv')
    df_sick_test.to_csv(f'sick_results/sick.fold.00000{fold}.test.csv')


