import pandas as pd

def get_n_best_results(file_path: str, n=1):
    df = pd.read_csv(file_path)
    df['k_to_schools_ratio'] = df['k'] / df['num_schools']
    grouped = df.groupby('experiment_number').apply(lambda x: x.nlargest(n, 'average_utility'))
    grouped = grouped.reset_index(drop=True)
    result = grouped[['experiment_number', 'num_students', 'num_schools', 'epsilon', 'algorithm', 'k', 'average_utility', 'k_to_schools_ratio']]
    result['k_to_schools_ratio'] = result['k_to_schools_ratio'].round(3)

    best_k_to_schools_ratio_mean = result['k_to_schools_ratio'].mean()

    return result, best_k_to_schools_ratio_mean


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    file_path = './data_out/new_experiment_results_grouped_ex_k_3.csv'
    top_results, best_k_to_schools_ratio_mean = get_n_best_results(file_path=file_path, n=1)
    print(top_results)
    print(best_k_to_schools_ratio_mean)