#from preprocessing import preprocess, Load_Sick, Load_Grid
from utils import callKNNs
from preprocessing import preprocess_sick, preprocess_grid
import pandas as pd

def load_ds(name, num_folds=10):

    folds_data = []

    for fold in range(num_folds):

        train_file = f'{name}_csv/{name}.fold.00000{fold}.train.csv'
        test_file = f'{name}_csv/{name}.fold.00000{fold}.test.csv'

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        folds_data.append((X_train, X_test, y_train, y_test))

    return folds_data

def preprocess():
    preprocess_sick()
    preprocess_grid() # Loads tiff and saves as csv

def main():
    print("MAIN")
    datasets = ['sick', 'grid']

    for ds in datasets:
        print(f"Executing dataset {ds}")
        for i, (X_train, X_test, y_train, y_test) in enumerate(load_ds(ds)): # Loads csv
            print(f"Fold {i}")
            callKNNs(X_train, X_test, y_train, y_test, ds, i)

if __name__ == "__main__":
    main()