import os
import sys

import pandas as pd

from knn import callKNN, callKNNs
from svm import callSVM, callSVMs
from preprocessing import preprocess_sick, preprocess_grid


def get_user_choice(prompt, options, is_numeric = False):
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            if is_numeric:
                print(f"  {option}")
            else:
                print(f" {i}. {option}")
        choice = input("Please enter the number of your choice: ")

        if choice in options or (is_numeric and int(choice) in options):
            return choice
        if not is_numeric and choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid choice. Try again.\n")

def loading_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '#' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r[{bar}] {percent:.2%} Complete')
    sys.stdout.flush()

def load_ds(name, num_folds=10):

    folds_data = []

    for fold in range(num_folds):

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"{name}_csv")
        train_file = os.path.join(base_dir, f'{name}.fold.00000{fold}.train.csv')
        test_file = os.path.join(base_dir, f'{name}.fold.00000{fold}.test.csv')

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

def runAllKNN():
    datasets = ['sick']  # ['sick', 'grid']
    model = 'knn'# 'svm'

    for ds in datasets:
        print(f"Executing dataset {ds}")
        for i, (X_train, X_test, y_train, y_test) in enumerate(load_ds(ds)):  # Loads csv
            print(f"Fold {i}")
            if model == 'knn':
                callKNNs(X_train, X_test, y_train, y_test, ds, i)
            elif model == 'svm':
                callSVMs(X_train, X_test, y_train, y_test, ds, i)

def startProgram():
    print("Welcome to our KNN and SVM application.")

    dataset = get_user_choice("Please, select the dataset you would like to use:", ["sick", "grid"])
    alg = get_user_choice("Please, select the algorithm to use:", ["knn", "svm"])

    if alg == 'svm':
        kernel = get_user_choice("Please, select the kernel to use:", ['rbf', 'sigmoid'])

        print("Computing the SVM with the specified parameters:")
        results = pd.DataFrame()
        for i, (X_train, X_test, y_train, y_test) in enumerate(load_ds(dataset)):
            res = callSVM(X_train, X_test, y_train, y_test, kernel)
            results = pd.concat([results, res], ignore_index=True)
            loading_bar(i+1, 10)

        print("\nComputation complete!")
        print(results.iloc[:, -9:].mean())

    elif alg == 'knn':
        k = get_user_choice("Please, select which K to use:", [1,3,5,7], True)
        dist_func = get_user_choice("Please, select a distance function to use:", ['minkowski1','minkowski2','HEOM'])
        voting_scheme = get_user_choice("Please, select a voting scheme to use :", ['Majority_class','Inverse_Distance_Weights', 'Sheppards_Work'])
        weight_scheme = get_user_choice("Please, select a weight scheme to use:", ['Mutual_classifier','Relief','ANOVA'])

        print(f"{k} - {dist_func} - {voting_scheme} - {weight_scheme}")
        print("Computing the KNN with the specified parameters:")
        results = pd.DataFrame()
        for i, (X_train, X_test, y_train, y_test) in enumerate(load_ds(dataset)):
            res = callKNN(X_train, X_test, y_train, y_test, dist_func, voting_scheme, weight_scheme, int(k))
            results = pd.concat([results, res], ignore_index=True)
            loading_bar(i+1, 10)

        print("\nComputation complete!")
        print(results.iloc[:, -9:].mean())

def main():
    ans = "y"
    while ans.lower() == "y":
        startProgram()
        ans = input("Do you want to do another test? (y/n)")

        while ans != "y" and ans != "n":
            print("Invalid choice. Try again.")
            ans = input("Do you want to do another test? (y/n)")


if __name__ == "__main__":
    main()