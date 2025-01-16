import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import random

from utils import (generate_random_profiles, generate_school_capacities, generate_k_restricted_preferences,
                   calculate_utility, calculate_utilities_from_prob, generate_unassigned_statistic, group_test_results)
from algorithm import k_boston_algorithm, k_gs_algorithm, manipulation_algorithm, algorithm_sampler
from data_analysis import get_n_best_results


# Допилить отдельные полезности для манипуляторов / честных игроков, мб отдельную статистику по unassigned_students для них
def run_experiment_k(algorithm: str,
                     num_students: int,
                     num_schools: int,
                     profiles: np.ndarray,
                     capacities: np.ndarray,
                     num_repeat_sampler: int,
                     # num_repeat_for_k: int,
                     k: int,
                     epsilon: float,
                     fair_indices: np.ndarray,
                     num_manipulations: int
                     ):
    # На данный момент считается, что boston только для k = num_schools
    # Иначе, возможно, надо будет другую схему манипуляций для boston

    # if algorithm == 'boston':
    #     k = num_schools

    if algorithm == 'boston':
        k = num_schools
        preferences = generate_k_restricted_preferences(profiles, k)
        manipulators = [0 for _ in range(num_students)]

    elif algorithm == 'gs':
        # Возможно, тут надо еще и это много раз повторять, если внутри manipulation_algorithm есть случайный выбор манипуляции/человека
        preferences, manipulators = manipulation_algorithm(algorithm=algorithm,
                                                           num_students=num_students,
                                                           num_schools=num_schools,
                                                           profiles=profiles,
                                                           capacities=capacities,
                                                           k=k,
                                                           epsilon=epsilon,
                                                           fair_indices=fair_indices,
                                                           num_manipulations=num_manipulations)
    else:
        raise ValueError('Algorithm must be either "boston" or "gs"')

    probabilities, unassigned_statistic = algorithm_sampler(algorithm=algorithm,
                                                                              num_students=num_students,
                                                                              num_schools=num_schools,
                                                                              preferences=preferences,
                                                                              capacities=capacities,
                                                                              k=k,
                                                                              num_repeat=num_repeat_sampler)

    utilities = calculate_utilities_from_prob(num_students=num_students,
                                              num_schools=num_schools,
                                              probabilities=probabilities,
                                              profiles=profiles)

    return probabilities, utilities, manipulators, unassigned_statistic


def run_experiment(num_students: int,
                   num_schools: int,
                   capacities: np.ndarray,
                   capacities_generated: bool,
                   num_capacities: int,
                   num_repeats_profiles: int = 10,
                   num_repeat_sampler: int = 1000,
                   epsilon: float = 0.1,
                   manipulators_ratio: float = 1.0,
                   num_manipulations: int = 3  # 3?
                   ):
    if num_students < num_schools:
        raise ValueError("Number of students cannot be less than the number of schools.")

    start_time = time.time()

    num_fair = round(num_students * (1 - manipulators_ratio))
    fair_indices = np.random.choice(num_students, num_fair, replace=False)

    experiment_results = []

    k_boston = num_schools

    for profile_number in range(num_repeats_profiles):
        profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)

        # Если доделать boston для k < num_schools, то добавить цикл для boston (и поменять ужас сколько индексов),
        # а внутри run_experiment_k переделать отработку boston
        # boston algorithm
        probabilities, utilities, manipulators, unassigned_statistic = run_experiment_k(
            algorithm="boston",
            num_students=num_students,
            num_schools=num_schools,
            profiles=profiles,
            capacities=capacities,
            num_repeat_sampler=num_repeat_sampler,
            k=k_boston,
            epsilon=epsilon,
            fair_indices=fair_indices,
            num_manipulations=num_manipulations)

        (average_percentage_unassigned_students,
         average_percentage_unassigned_fair_students,
         average_percentage_unassigned_manipulator_students,
         average_utility_fair_students,
         average_utility_manipulator_students) = generate_unassigned_statistic(num_students=num_students,
                                                                               fair_indices=fair_indices,
                                                                               unassigned_statistic=unassigned_statistic,
                                                                               utilities=utilities)

        new_row_boston = {"num_students": [num_students],
                          "num_schools": [num_schools],
                          "average_runtime": [time.time() - start_time],
                          "capacities": [capacities],
                          "capacities_generated": [capacities_generated],
                          "num_capacities": [num_capacities],
                          "num_repeats_profiles": [num_repeats_profiles],
                          "num_repeat_sampler": [num_repeat_sampler],
                          "epsilon": [epsilon],
                          "manipulators_ratio": [manipulators_ratio],
                          "default_fair_num_student": [round(num_students * (1 - manipulators_ratio))],  # Можно убрать
                          "num_manipulations": [num_manipulations],
                          "algorithm": ["boston"],
                          "k": [k_boston],
                          "probabilities": [probabilities],
                          "utilities": [utilities],
                          "average_utility": [np.sum(utilities) / num_students],
                          "average_number_manipulations": [np.sum(manipulators)],
                          "possible_percentage_manipulators": [
                              (num_students - round(num_students * (1 - manipulators_ratio))) / num_students * 100],
                          "average_actual_percentage_manipulators": [np.count_nonzero(manipulators) / num_students * 100],
                          "average_percentage_unassigned_students": [average_percentage_unassigned_students],
                          "average_percentage_unassigned_fair_students": [average_percentage_unassigned_fair_students],
                          "average_percentage_unassigned_manipulator_students": [average_percentage_unassigned_manipulator_students],
                          "average_utility_fair_students": [average_utility_fair_students],
                          "average_utility_manipulator_students": [average_utility_manipulator_students]
                          }

        boston_df = pd.DataFrame(new_row_boston, index=[0])
        experiment_results.append(boston_df)

        # gs algorithm
        for k_gs in range(1, num_schools + 1):
            probabilities, utilities, manipulators, unassigned_statistic = run_experiment_k(
                algorithm="gs",
                num_students=num_students,
                num_schools=num_schools,
                profiles=profiles,
                capacities=capacities,
                num_repeat_sampler=num_repeat_sampler,
                k=k_gs,
                epsilon=epsilon,
                fair_indices=fair_indices,
                num_manipulations=num_manipulations)

            (average_percentage_unassigned_students,
             average_percentage_unassigned_fair_students,
             average_percentage_unassigned_manipulator_students,
             average_utility_fair_students,
             average_utility_manipulator_students) = generate_unassigned_statistic(num_students=num_students,
                                                                                   fair_indices=fair_indices,
                                                                                   unassigned_statistic=unassigned_statistic,
                                                                                   utilities=utilities)

            new_row_gs = {"num_students": [num_students],
                          "num_schools": [num_schools],
                          "average_runtime": [time.time() - start_time],
                          "capacities": [capacities],
                          "capacities_generated": [capacities_generated],
                          "num_capacities": [num_capacities],
                          "num_repeats_profiles": [num_repeats_profiles],
                          "num_repeat_sampler": [num_repeat_sampler],
                          "epsilon": [epsilon],
                          "manipulators_ratio": [manipulators_ratio],
                          "default_fair_num_student": [round(num_students * (1 - manipulators_ratio))],  # Можно убрать
                          "num_manipulations": [num_manipulations],
                          "algorithm": ["gs"],
                          "k": [k_gs],
                          "probabilities": [probabilities],
                          "utilities": [utilities],
                          "average_utility": [np.sum(utilities) / num_students],
                          "average_number_manipulations": [np.sum(manipulators)],
                          "possible_percentage_manipulators": [
                              (num_students - round(num_students * (1 - manipulators_ratio))) / num_students * 100],
                          "average_actual_percentage_manipulators": [np.count_nonzero(manipulators) / num_students * 100],
                          "average_percentage_unassigned_students": [average_percentage_unassigned_students],
                          "average_percentage_unassigned_fair_students": [average_percentage_unassigned_fair_students],
                          "average_percentage_unassigned_manipulator_students": [average_percentage_unassigned_manipulator_students],
                          "average_utility_fair_students": [average_utility_fair_students],
                          "average_utility_manipulator_students": [average_utility_manipulator_students]
                          }

            gs_df = pd.DataFrame(new_row_gs, index=[0])
            experiment_results.append(gs_df)

    return experiment_results


def massive_run(tests: list, display_progress: bool = False):
    # Принимает список экспериментов, запускает их параллельно, формирует pd.df
    exp_num = 0

    test_results = pd.DataFrame(columns=["num_students", "num_schools", "average_runtime", "capacities", "capacities_generated",
                                         "num_capacities", "num_repeats_profiles",
                                         "num_repeat_sampler", "epsilon", "manipulators_ratio", "default_fair_num_student",
                                         "num_manipulations",
                                         "algorithm", "k", "probabilities", "utilities", "average_utility",
                                         "average_utility_fair_students", "average_utility_manipulator_students",
                                         "possible_percentage_manipulators", "average_actual_percentage_manipulators",
                                         "average_number_manipulations",
                                         "average_percentage_unassigned_students", "average_percentage_unassigned_fair_students",
                                         "average_percentage_unassigned_manipulator_students",
                                         ])

    if display_progress:
        iterator = tqdm(tests, desc="Выполнение тестов")
    else:
        iterator = tests

    for exp_number, test in iterator:
        exp_num += 1

        params = {
            'num_students': test['num_students'],
            'num_schools': test['num_schools']
        }

        for key in ['num_repeats_profiles', 'num_repeat_sampler', 'epsilon', 'manipulators_ratio', 'num_manipulations']:
            if key in test:
                params[key] = test[key]

        if 'capacities' in test:
            params['capacities'] = test['capacities']
            params['capacities_generated'] = False
            params['num_capacities'] = 1

            experiment_results = run_experiment(**params)
            for df in experiment_results:
                df.insert(0, 'experiment_number', int(exp_number) + 1)
            experiment_results.append(test_results)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                test_results = pd.concat(experiment_results, ignore_index=True)
            # test_results = pd.concat([test_results, experiment_results], ignore_index=True)

        else:
            params['capacities_generated'] = True
            if 'num_capacities' in test:
                params['num_capacities'] = test['num_capacities']
            else:
                params['num_capacities'] = 1
            for capacities_index in range(params['num_capacities']):
                capacities = generate_school_capacities(num_students=test['num_students'], num_schools=test['num_schools'])
                params['capacities'] = capacities

                experiment_results = run_experiment(**params)
                for df in experiment_results:
                    df.insert(0, 'experiment_number', int(exp_number) + 1)
                experiment_results.append(test_results)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    test_results = pd.concat(experiment_results, ignore_index=True)

    return test_results


def parallel_run(tests: list, batch_size: int = 1, n_jobs: int = 1, display_progress: bool = False):
    tests = list(enumerate(tests))
    random.shuffle(tests)

    num_batch = int(len(tests) / batch_size) if len(tests) % batch_size == 0 else int(len(tests) / batch_size) + 1
    tests_with_batch = [tests[i * batch_size: (i + 1) * batch_size] for i in range(num_batch)]

    with Parallel(n_jobs=n_jobs, verbose=11, backend='threading') as parallel:
        results = parallel(
            delayed(massive_run)(test_batch, display_progress=display_progress) for test_batch in tests_with_batch)

    test_results = pd.concat(results, ignore_index=True)

    return test_results


if __name__ == '__main__':

    num_students = 20
    num_schools = 8
    profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
    capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
    num_repeat_sampler = 100
    k = 2
    epsilon = 0.1
    num_manipulations = 5

    # tests = [{"num_students": 20, "num_schools": 8, "capacities": np.array([8, 4, 3, 1, 1, 1, 1, 1]), "num_capacities": 3, "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5}]
    tests = [{"num_students": 20, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 6},
             {"num_students": 10, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 15, "num_schools": 6, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 10, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 18, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 5},
             {"num_students": 20, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 7},
             {"num_students": 15, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "manipulators_ratio": 0.6, "num_manipulations": 4}]

    tests = [{"num_students": 20, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 6},
             {"num_students": 10, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 15, "num_schools": 6, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 10, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 18, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 5},
             {"num_students": 20, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 7},
             {"num_students": 15, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 4}]

    tests = [{"num_students": 20, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 6},
             {"num_students": 10, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 15, "num_schools": 6, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 10, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 18, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 20, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 7},
             {"num_students": 15, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 4},
             {"num_students": 24, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 18, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 2},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 5,
              "num_repeat_sampler": 50, "epsilon": 0.005, "num_manipulations": 3}
             ]

    tests = [{"num_students": 20, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 6},
             {"num_students": 10, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 10},
             {"num_students": 15, "num_schools": 6, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 10, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 7, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 18, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 5},
             {"num_students": 20, "num_schools": 10, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 7},
             {"num_students": 15, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 4},
             {"num_students": 24, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 18, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 2},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 40, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.01, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 3},
             {"num_students": 40, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 3},
             {"num_students": 50, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 8, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 5},
             {"num_students": 10, "num_schools": 2, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 1},
             {"num_students": 26, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.005, "num_manipulations": 3},
             {"num_students": 28, "num_schools": 4, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 3},
             {"num_students": 30, "num_schools": 6, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 3},
             {"num_students": 32, "num_schools": 5, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 3},
             {"num_students": 18, "num_schools": 3, "num_capacities": 5, "num_repeats_profiles": 10,
              "num_repeat_sampler": 100, "epsilon": 0.02, "num_manipulations": 3},
             ]

    pd.set_option('display.max_columns', None)

    experiment_results = parallel_run(tests, batch_size=1, n_jobs=-2, display_progress=False)

    experiment_results = experiment_results[
        ['experiment_number'] + [col for col in experiment_results.columns if col != 'experiment_number']]
    # experiment_results = experiment_results.sort_values(by=['experiment_number', 'k', 'algorithm'])
    experiment_results_grouped = group_test_results(experiment_results)

    # print(experiment_results)

    # experiment_results.to_csv('./data_out/new_experiment_results_ex.csv', index=False)
    file_path = './data_out/new_experiment_results_grouped_ex_k_3.csv'
    experiment_results_grouped.to_csv(path_or_buf=file_path, index=False)
    print(get_n_best_results(file_path=file_path, n=2))
