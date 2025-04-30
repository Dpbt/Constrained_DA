import os
import random
import pandas as pd
import numpy as np

from test_system import run_full_tests


random.seed(42)
np.random.seed(42)

# params_lists = {
#     "num_students": [150],
#     "num_schools": [5, 10, 20],
#     "num_capacities": [5],
#     "num_repeats_profiles": [5],
#     "num_repeat_sampler": [50],
#     "epsilon": [0.002, 0.005, 0.01, 0.02],
#     "manipulators_ratio": [0.25, 0.5, 0.75, 1],
#     "num_manipulations_ratio": [0.25, 0.5, 0.75, 1],
# }

# params_lists = {
#     "num_students": [100, 200, 300, 400, 500],
#     "num_schools": [5, 10, 20],
#     "num_capacities": [5],
#     "num_repeats_profiles": [5],
#     "num_repeat_sampler": [50],
#     "epsilon": [0.002, 0.005, 0.01],
#     "manipulators_ratio": [0.5, 0.75, 1],
#     "num_manipulations_ratio": [0.5, 0.75, 1],
# }

params_lists = {
    "num_students": [100],
    "num_schools": [2, 5, 8, 11, 14],
    "num_capacities": [5],
    "num_repeats_profiles": [5],
    "num_repeat_sampler": [50],
    "epsilon": [0.001, 0.002, 0.005, 0.01],
    "manipulators_ratio": [0.25, 0.5, 0.75, 1.0],
    "num_manipulations_ratio": [0.5, 0.75, 1.0],
}

params_lists = {
    "num_students": [12, 14],
    "num_schools": [3, 5, 8],
    "num_capacities": [5],
    "num_repeats_profiles": [5],
    "num_repeat_sampler": [50],
    "epsilon": [0.001],
    "manipulators_ratio": [0.75, 1.0],
    "num_manipulations_ratio": [0.75, 1.0],
}

experiment_results = run_full_tests(params_lists=params_lists,
                                    batch_size=1,
                                    n_jobs=-2,
                                    display_progress=True,
                                    save_path='./data_out/data_out_new_2.csv',
                                    print_n_best_results=True,
                                    )



# tests = generate_tests_from_lists(**params_lists)
#
# files = os.listdir('./data_out')
#
# exp_numbers = []
# # for f in files:
# #     if f.startswith('test_results_') and f.endswith('.csv'):
# #         number = f.split('_')[-1].replace('.csv', '')
# #         exp_numbers.append(int(number))
#
# tests = [(i, test) for i, test in enumerate(tests) if i not in exp_numbers]
#
# print(len(tests))
# print(tests)
#
# pd.set_option('display.max_columns', None)
#
# random.seed(42)
# np.random.seed(42)
#
# experiment_results = parallel_run_old(tests, batch_size=1, n_jobs=-2, display_progress=False)
#
# experiment_results = experiment_results[
#     ['experiment_number'] + [col for col in experiment_results.columns if col != 'experiment_number']]
# experiment_results_grouped = group_test_results(experiment_results)
#
# file_path = './data_out/data_out_old.csv'
# experiment_results_grouped.to_csv(path_or_buf=file_path, index=False)
# print(get_n_best_results(file_path=file_path, n=1))


