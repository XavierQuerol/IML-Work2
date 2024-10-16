#from preprocessing import preprocess, Load_Sick, Load_Grid
from utils import callKNNs


def preprocess():
    datasets = ['Sick', 'Grid']
    preprocess_datasets(datasets) # Loads tiff and saves as csv

def main():
    for cv in Load_Sick(): # Loads csv
       X_train, X_test, y_train, y_test = cv
       callKNNs(X_train, X_test, y_train, y_test)

    for cv in Load_Grid():
       X_train, X_test, y_train, y_test = cv
       callKNNs(X_train, X_test, y_train, y_test)