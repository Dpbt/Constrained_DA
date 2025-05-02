import random

import numpy as np

random.seed(42)
np.random.seed(42)

from test_system import run_full_tests

# params_lists which was used to retrieve the data_out/data_out_server.csv file
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

# params_lists which was used to retrieve the data_out/data_out_100.csv file
# params_lists = {
#     "num_students": [100],
#     "num_schools": [2, 5, 8, 11, 14],
#     "num_capacities": [5],
#     "num_repeats_profiles": [5],
#     "num_repeat_sampler": [50],
#     "epsilon": [0.001, 0.002, 0.005, 0.01],
#     "manipulators_ratio": [0.25, 0.5, 0.75, 1.0],
#     "num_manipulations_ratio": [0.5, 0.75, 1.0],
# }

# params_lists that can be used for a quick work example
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
                                    save_path='./data_out/data_out_new_simulation.csv',
                                    print_n_best_results=True,
                                    )