
### Team
- Aina Llaneras
- Maria Angelica Jaimes
- Xavier Querol 
- Bernat Comas


### Running script for the first time

1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```

Execute main.py by running:
```bash
 python3 main.py
 ```

Then a menu will be displayed with the instructions to execute any combination of hyperparameters. Follow the instructions
- If you put in a wrong parameter the app will make you input it again.
- You can put both the number in the left side of the parameter and the name of the parameter itself as shown in the options.

If you want to run all the KNN options you can choose to do it one by one through the main menu or by executing the function runAllKNN present in the file main.py. We recommend the first option because running everything is a very long process.


Execute evaluation.py to validate differences among the best models and evaluate the reduction effect. Results are precomputed.
```bash
 python3 evaluation.py
 ```


Execute graphics.py to show the graphics included in the report. Results are precomputed.
```bash
 python3 graphics.py
 ```

 ### Folder structure

    ├── code                   # .py files
    ├── documentation          # Documentation about reduction methods
    ├── grid                   # Source files about grid dataset
    ├── grid_csv               # Generated files about grid dataset
    ├── sick                   # Source files about sick dataset
    ├── sick_csv               # Generated files about sick dataset
    ├── results_knn            # Results of knn algorithm
    ├── results_knn_reduced    # Results of knn algorithm after reductions
    ├── results_svm            # Results of svm algorithm
    ├── results_svm_reduced    # Results of svm algorithm after reductions
    └── README.md