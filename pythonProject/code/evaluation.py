from autorank import autorank, plot_stats, create_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import scikit_posthocs as sp


def read_csv(folder_path, dataset_name, reduction_mode):
    if reduction_mode:
        csv_files = [f for f in os.listdir(folder_path) if (dataset_name in f and reduction_mode in f and 'all' not in f)]
    else:
        csv_files = [f for f in os.listdir(folder_path) if (dataset_name in f and 'all' not in f)]
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined


def aggregate_metrics(df_combined, method):
    if method == 'knn':
        columns = ['K', 'Distance', 'Voting scheme', 'Weight scheme']
    elif method == 'svm':
        columns = ['Kernel']
    
    grouped_df = df_combined.groupby(columns)

    # Compute mean and standard deviation of the relevant metrics
    metrics_summary = grouped_df.agg({
        'Accuracy': ['mean', 'std'],
        'Precision_Class_0': ['mean', 'std'],
        'Recall_Class_0': ['mean', 'std'],
        'F1_Class_0': ['mean', 'std'],
        'Precision_Class_1': ['mean', 'std'],
        'Recall_Class_1': ['mean', 'std'],
        'F1_Class_1': ['mean', 'std'],
        'Solving Time': ['mean', 'std']
    }).reset_index()


    # Rename the columns for clarity
    metrics_summary.columns = columns + [
                            'Accuracy_mean', 'Accuracy_std',
                            'Precision_Class_0_mean', 'Precision_Class_0_std',
                            'Recall_Class_0_mean', 'Recall_Class_0_std',
                            'F1_Class_0_mean', 'F1_Class_0_std',
                            'Precision_Class_1_mean', 'Precision_Class_1_std',
                            'Recall_Class_1_mean', 'Recall_Class_1_std',
                            'F1_Class_1_mean', 'F1_Class_1_std',
                            'Solving Time_mean', 'Solving Time_std']

    metrics_summary = metrics_summary.sort_values(by='Accuracy_mean', ascending = False)

    # Get the best hyperparameters (the first row after sorting)
    best_hyperparams = metrics_summary.iloc[0][columns].to_dict()
    best_accuracy = metrics_summary.iloc[0]['Accuracy_mean']

    # Print the best hyperparameters and their accuracy
    print("Best Hyperparameters:")
    print(best_hyperparams)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return metrics_summary, best_hyperparams


def get_metrics_knn(best_models, df_combined, metric):
    metric_values = []
    for _, row in best_models.iterrows():
        # Store model identification (e.g., hyperparameters)
        model_id = (row['Voting scheme'], row['Weight scheme'], row['Distance'], row['K'])

        # Filter for the current hyperparameter combination
        filtered_df = df_combined[
            (df_combined['Voting scheme'] == row['Voting scheme']) & 
            (df_combined['Weight scheme'] == row['Weight scheme']) &
            (df_combined['Distance'] == row['Distance']) &
            (df_combined['K'] == row['K'])
        ]
        # Collect accuracy values for this combination
        metric_values.append(list(filtered_df[metric].values))
    
    return metric_values


def get_metrics_svm(best_models, df_combined, metric):
    metric_values = []
    for _, row in best_models.iterrows():
        # Store model identification (e.g., hyperparameters)
        model_id = (row['Kernel'])

        # Filter for the current hyperparameter combination
        filtered_df = df_combined[
            (df_combined['Kernel'] == row['Kernel'])
        ]
    # Collect accuracy values for this combination
        metric_values.append(list(filtered_df[metric].values))
    
    return metric_values


def plot_violin(data, metric, ax):
    """Helper function to plot the violin and strip plot."""
    sns.violinplot(data=data, ax=ax, inner=None, palette="pastel", linewidth=1.5)
    sns.stripplot(data=data, ax=ax, color='black', alpha=0.7, jitter=True, size=3)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f"Model {metric} Values", fontsize=16, fontweight='bold')
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xlabel("Models", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)


def plot_test(data, metric, results=None, autorank=False):
    if autorank:
        # Create a figure with two subplots for autorank and violin plot
        fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [2, 5]})
        
        # Plot autorank results in the first subplot
        plot_stats(results, ax=axs[0])
        axs[0].set_title("Autorank Results")
        
        # Plot the violin in the second subplot
        plot_violin(data, metric, axs[1])
        
    else:
        # Create a single figure for the violin plot if autorank is False
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_violin(data, metric, ax)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def evaluation_test_autorank(data, metric, p_value=0.05, plot=False):
    if metric == 'Accuracy':
        order = 'descending'
    elif metric == 'Solving Time':
        order = 'ascending'
    results = autorank(data, alpha = p_value, order = order)
    try:
        create_report(results)
    except:
        pass
    if plot:
        plot_test(data, metric, results=results, autorank=True)

            
def evaluation_test(data, metric, p_value=0.05, plot=False):

    metric_values = [list(model) for model in data.T.values]

    # Perform the Friedman test if there are three or more models
    if len(metric_values) >= 3:
        stat, p_value_friedman = stats.friedmanchisquare(*metric_values)
        print(f"Friedman test statistic: {stat}, p-value: {p_value_friedman}")

        if p_value_friedman < p_value:
            print("Significant differences found, conducting Nemenyi post-hoc test")
            data = pd.DataFrame(metric_values).T  # Transpose so each column is a model
            nemenyi_result = sp.posthoc_nemenyi_friedman(data)
            print(nemenyi_result)
        else:
            print("No significant differences found between the models.")

    elif len(metric_values) == 2:
        stat, p_value = stats.wilcoxon(metric_values[0], metric_values[1])
        print(f"Wilcoxon signed-rank test statistic: {stat}, p-value: {p_value}")

        if p_value < 0.05:
            print("Significant difference between the two models.")
        else:
            print("No significant differences found between the two models.")

    else:
        print("Not enough data to perform the test.")
        return
    if plot:
        plot_test(data, metric, results=None, autorank=False)


def evaluate_model(dataset_name, method, metric, type_evaluation= 'our_criteria', plot=False):

    votings = {'Majority_class': 'Mj', 'Inverse_Distance_Weights': 'IDW','Sheppards_Work': 'SW'}
    distances = {'minkowski1': 'm1', 'minkowski2': 'm2', 'HEOM': 'H'}
    weighting = {'Mutual_classifier':'MC', 'Relief': 'R', 'ANOVA': 'A'}

    df_combined = read_csv(f'results_{method}', dataset_name, False)
    metrics_summary, best_hyperparams = aggregate_metrics(df_combined, method)
    metrics_summary.to_csv(f'results_{method}/results_{dataset_name}_all.csv', index=False)

    if method == 'knn':
        best_models = metrics_summary.iloc[[0, 1, 2, 3, 4 ,90, 100]]
        metric_values = get_metrics_knn(best_models, df_combined, metric)
        data = pd.DataFrame()
        for el, (i,row) in zip(metric_values, best_models.iterrows()):
            data[f"{votings[row['Voting scheme']]}_{weighting[row['Weight scheme']]}_{distances[row['Distance']]}_{row['K']}"] = el

    elif method == 'svm':
        best_models = metrics_summary.iloc[[0, 1]]
        metric_values = get_metrics_svm(best_models, df_combined ,metric)
        data = pd.DataFrame()
        for el, (i,row) in zip(metric_values, best_models.iterrows()):
            data[f"{row['Kernel']}"] = el
            
    if type_evaluation == 'autorank':
        evaluation_test_autorank(data, metric, 0.05, plot=plot)
    elif type_evaluation == 'our_criteria':
        evaluation_test(data, metric, 0.05, plot=plot)


def evaluate_all_models(type_evaluation='our_criteria', plot=False):
    dataset_names = ['grid','sick']
    methods = ['knn','svm']
    for dataset_name in dataset_names:
        for method in methods:
            for metric in ['Accuracy', 'Solving Time']:
                print(f'Evaluating method {method} on dataset {dataset_name} and metric {metric}')
                evaluate_model(dataset_name, method, metric, type_evaluation= type_evaluation, plot=plot)

def evaluation_t_test(data, metric, p_value, plot=False):
    metric_values = [list(model) for model in data.T.values]

    # Perform paired t-test
    stat, p_value_ttest = stats.ttest_rel(metric_values[0], metric_values[1])
    print(f"Paired t-test statistic: {stat}, p-value: {p_value}")

    if p_value_ttest < p_value:
        print("Significant difference found between the two models.")
    else:
        print("No significant differences found between the two models.")
    if plot:
        plot_test(data, metric, results=None, autorank=False)

def calculate_statistics(data, metric):
    """Calculate mean and std as formatted string 'mean ± std'."""
    mean_val = data[metric].mean()
    std_val = data[metric].std()
    return f"{mean_val:.4f} ± {std_val:.4f}"

def evaluate_reduction(dataset_name, method, reduction_method, type_evaluation, summary_statistics = None, plot = False):

    best_params = {'grid': {
                        'knn': {
                            'K': 7,
                            'Distance': 'HEOM',
                            'Voting scheme': 'Inverse_Distance_Weights',
                            'Weight scheme': 'Mutual_classifier'},
                        'svm': 
                            {'Kernel': 'rbf'}}, 
                    'sick': {
                        'knn':{
                            'K': 7, 
                            'Distance': 'minkowski1', 
                            'Voting scheme': 'Majority_class', 
                            'Weight scheme': 'ANOVA'},
                        'svm': 
                            {'Kernel': 'rbf'}}}
        
    num_samples = {'grid': 1700, 'sick': 3395}
    
    # Load original data
    results = read_csv(f'results_{method}', dataset_name, False)
    # Filter original data based on best parameters
    params = best_params[dataset_name][method]
    for key, value in params.items():
        results = results[results[key] == value]

    results_reduced = read_csv(f'results_{method}_reduced', dataset_name, reduction_method)

    data_accuracy = pd.DataFrame({
        'Original': list(results['Accuracy']), 
        'Reduced': list(results_reduced['Accuracy'])
    })
    data_solving_time = pd.DataFrame({
        'Original': list(results['Solving Time']), 
        'Reduced': list(results_reduced['Solving Time'])
    })

    if type_evaluation == 'our_criteria':
        print('Evaluating Accuracy')
        evaluation_t_test(data_accuracy, 'Accuracy', 0.05, plot=plot)
        print('Evaluating Solving Time')
        evaluation_t_test(data_solving_time, 'Solving Time', 0.05, plot=plot)
    else:
        print('Evaluating Accuracy')
        evaluation_test_autorank(data_accuracy, 'Accuracy', 0.05, plot=plot)
        print('Evaluating Solving Time')
        evaluation_test_autorank(data_solving_time, 'Solving Time', 0.05, plot=plot)

    if summary_statistics is not None:
        # Calculate statistics for both metrics in the original data
        original_accuracy = calculate_statistics(results, 'Accuracy')
        original_solving_time = calculate_statistics(results, 'Solving Time')
       # Add the original result row without reduction for both metrics
        summary_statistics = pd.concat([
            summary_statistics,
            pd.DataFrame({
                'Dataset': [dataset_name],
                'Method': [method],
                'Reduction': ['None'],
                'Accuracy': [original_accuracy],
                'Solving Time': [original_solving_time],
                'Num Samples': [num_samples[dataset_name]]
            })
        ], ignore_index=True)

        # Now load reduced data for this reduction method
        
        reduced_accuracy = calculate_statistics(results_reduced, 'Accuracy')
        reduced_solving_time = calculate_statistics(results_reduced, 'Solving Time')
        reduced_num_samples = calculate_statistics(results_reduced, 'Num samples')

        # Add the reduced result row for both metrics
        summary_statistics = pd.concat([
            summary_statistics,
            pd.DataFrame({
                'Dataset': [dataset_name],
                'Method': [method],
                'Reduction': [reduction_method],
                'Accuracy': [reduced_accuracy],
                'Solving Time': [reduced_solving_time],
                'Num Samples': [reduced_num_samples]
            })
        ], ignore_index=True)

    if summary_statistics is not None:
        return summary_statistics

def evaluate_all_reductions(type_evaluation, plot=False):
    dataset_names = ['grid','sick']
    methods = ['knn','svm']
    summary_statistics = pd.DataFrame()
    for dataset_name in dataset_names:
        for method in methods:
            for i, reduction in enumerate(['CNN', 'DROP', 'EENTh']):
                print(f'Evaluating method {method} on dataset {dataset_name} and reduction {reduction}')
                summary_statistics = evaluate_reduction(dataset_name, method, reduction, type_evaluation=type_evaluation, summary_statistics=summary_statistics, plot=plot)

    summary_statistics = summary_statistics.drop_duplicates()
    return summary_statistics