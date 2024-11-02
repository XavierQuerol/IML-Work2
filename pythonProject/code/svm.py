import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC

from code.utils import computeMetrics

def callSVMs(X_train, X_test, y_train, y_test, ds_name, fold):
    kernels = ['rbf', 'sigmoid']

    columns=['Kernel', 'Accuracy', 'Solving Time']
    classes = y_train.unique()

    for i in classes:
        columns.append(f'Precision_Class_{i}')
        columns.append(f'Recall_Class_{i}')
        columns.append(f'F1_Class_{i}')
    results = pd.DataFrame(columns=columns)

    for kernel in kernels:
        print(f" - Using kernel {kernel}")
        start = time.time()
        svm = SVC(kernel=kernel)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        solving_time = time.time() - start
        accuracy, precision, recall, f1 = computeMetrics(y_test, y_pred)
        res = {'Kernel': kernel, 'Accuracy': accuracy,'Solving Time': solving_time}
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            res[f'Precision_Class_{i}'] = p
            res[f'Recall_Class_{i}'] = r
            res[f'F1_Class_{i}'] = f

        new_row = pd.DataFrame([res])

        results = pd.concat([results.astype(new_row.dtypes), new_row.astype(results.dtypes)], ignore_index=True)

        print(f"This combination took {solving_time} seconds")

    results.to_csv(f'results_svm/results_{ds_name}_{fold}.csv', index=False)


def callSVM(X_train, X_test, y_train, y_test, kernel):

    columns=['Kernel', 'Accuracy', 'Solving Time', 'Num samples']
    classes = y_train.unique()

    for i in classes:
        columns.append(f'Precision_Class_{i}')
        columns.append(f'Recall_Class_{i}')
        columns.append(f'F1_Class_{i}')
    results = pd.DataFrame(columns=columns)

    start = time.time()
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    solving_time = time.time() - start
    accuracy, precision, recall, f1 = computeMetrics(y_test, y_pred)
    res = {'Kernel': kernel, 'Accuracy': accuracy,'Solving Time': solving_time, 'Num samples': len(X_train)}
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        res[f'Precision_Class_{i}'] = p
        res[f'Recall_Class_{i}'] = r
        res[f'F1_Class_{i}'] = f

    new_row = pd.DataFrame([res])

    results = pd.concat([results.astype(new_row.dtypes), new_row.astype(results.dtypes)], ignore_index=True)

    return results