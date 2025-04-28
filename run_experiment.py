import os
import pandas as pd

from test_system import parallel_run
from utils import generate_tests_from_lists, group_test_results
from data_analysis import get_n_best_results

random.seed(42)
np.random.seed(42)

# tests_lists = {
#     "num_students": [150],
#     "num_schools": [5, 10, 20],
#     "num_capacities": [5],
#     "num_repeats_profiles": [5],
#     "num_repeat_sampler": [50],
#     "epsilon": [0.002, 0.005, 0.01, 0.02],
#     "manipulators_ratio": [0.25, 0.5, 0.75, 1],
#     "num_manipulations": [0.25, 0.5, 0.75, 1],
# }

# tests_lists = {
#     "num_students": [100, 200, 300, 400, 500],
#     "num_schools": [5, 10, 20],
#     "num_capacities": [5],
#     "num_repeats_profiles": [5],
#     "num_repeat_sampler": [50],
#     "epsilon": [0.002, 0.005, 0.01],
#     "manipulators_ratio": [0.5, 0.75, 1],
#     "num_manipulations": [0.5, 0.75, 1],
# }

tests_lists = {
    "num_students": [100],
    "num_schools": [2, 5, 8, 11, 14],
    "num_capacities": [5],
    "num_repeats_profiles": [5],
    "num_repeat_sampler": [50],
    "epsilon": [0.001, 0.002, 0.005, 0.01],
    "manipulators_ratio": [0.25, 0.5, 0.75, 1],
    "num_manipulations": [0.5, 0.75, 1],
}

tests_lists = {
    "num_students": [20],
    "num_schools": [5],
    "num_capacities": [5],
    "num_repeats_profiles": [5],
    "num_repeat_sampler": [50],
    "epsilon": [0.001],
    "manipulators_ratio": [1],
    "num_manipulations": [1],
}

tests = generate_tests_from_lists(**tests_lists)

files = os.listdir('./data_out')

exp_numbers = []
# for f in files:
#     if f.startswith('test_results_') and f.endswith('.csv'):
#         number = f.split('_')[-1].replace('.csv', '')
#         exp_numbers.append(int(number))

tests = [[i, test] for i, test in enumerate(tests) if i not in exp_numbers]

print(len(tests))

pd.set_option('display.max_columns', None)

experiment_results = parallel_run(tests, batch_size=1, n_jobs=-2, display_progress=False)

experiment_results = experiment_results[
    ['experiment_number'] + [col for col in experiment_results.columns if col != 'experiment_number']]
experiment_results_grouped = group_test_results(experiment_results)

file_path = './data_out/data_out_lt.csv'
experiment_results_grouped.to_csv(path_or_buf=file_path, index=False)
print(get_n_best_results(file_path=file_path, n=1))
