import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from algorithm import algorithm_sampler, manipulation_algorithm
from analysis import get_n_best_results
from utils import (AlgorithmEnum, generate_random_profiles, generate_school_capacities,
                   generate_k_restricted_preferences, calculate_utilities_from_probs,
                   generate_unassigned_statistic, group_test_results, generate_tests_from_lists,
                   make_result_row_run_experiment)

SAVE_PATH_DEFAULT = "./data_out/data_out_default.csv"

pd.set_option('display.max_columns', None)


def run_experiment_k(
        algorithm: AlgorithmEnum,
        num_students: int,
        num_schools: int,
        profiles: np.ndarray,
        capacities: np.ndarray,
        num_repeat_sampler: int,
        k: int,
        epsilon: float,
        fair_indices: np.ndarray,
        num_manipulations: int
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Performs an experiment for the given parameters and a fixed constraint on the lengths of the preference lists k.
    First generates the final preference lists: for BOSTON_MECHANISM without manipulation, for K_GS_MECHANISM runs
    the list manipulation algorithm. Then algorithm_sampler is run on the resulting preference lists, repeating
    the execution of the mechanism on num_repeat_sampler lists once and averaging the resulting data.
    Then the average utility for all students on such lists is calculated.

    In test_system.py, the function is called from run_experiment().

    Parameters:
        algorithm (AlgorithmEnum): Matching mechanism to test.
                                   Available options: AlgorithmEnum.BOSTON_MECHANISM, AlgorithmEnum.K_GS_MECHANISM
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        profiles (np.ndarray): Matrix of cardinal utility of students from schools (shape: (num_students, num_schools))
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        num_repeat_sampler (int): Number of repetitions of the algorithm on generated preference lists
        k (int): Length of preference lists (k <= num_schools)
        epsilon (float): Minimum utility improvement threshold
        fair_indices (np.ndarray): Indices of non-manipulating students
        num_manipulations (int): Max manipulations per student (Calculated as round(num_manipulations_ratio * num_schools))

    Returns:
        Tuple containing:
        - probabilities (np.ndarray): Average assignment probabilities (shape: (num_students, num_schools))
        - utilities (np.ndarray): Average student utilities (shape: (num_students,))
        - manipulators (List[int]): Manipulation counters per student
        - unassigned_statistic (dict): Average statistics about unassigned students

    Raises:
        ValueError: For unsupported algorithms (only BOSTON_MECHANISM and K_GS_MECHANISM are supported now)
    """
    # Algorithm-specific initialization to generate preference lists
    if algorithm == AlgorithmEnum.BOSTON_MECHANISM:
        preferences = generate_k_restricted_preferences(profiles, num_schools)
        manipulators = [0 for _ in range(num_students)]

    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        preferences, manipulators = manipulation_algorithm(
            algorithm=AlgorithmEnum.K_GS_MECHANISM,
            num_students=num_students,
            num_schools=num_schools,
            profiles=profiles,
            capacities=capacities,
            k=k,
            epsilon=epsilon,
            fair_indices=fair_indices,
            num_manipulations=num_manipulations
        )
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}. "
                         f"Only AlgorithmEnum.K_GS_MECHANISM and AlgorithmEnum.BOSTON_MECHANISM are supported now.")

    # Sampling of algorithm results on generated preference lists
    probabilities, unassigned_statistic = algorithm_sampler(
        algorithm=algorithm,
        num_students=num_students,
        num_schools=num_schools,
        preferences=preferences,
        capacities=capacities,
        k=k,
        num_repeat=num_repeat_sampler
    )

    utilities = calculate_utilities_from_probs(
        num_schools=num_schools,
        probabilities=probabilities,
        profiles=profiles
    )

    return probabilities, utilities, manipulators, unassigned_statistic


def run_experiment(num_students: int,
                   num_schools: int,
                   capacities: np.ndarray,
                   capacities_generated: bool,
                   num_capacities: int,
                   num_repeats_profiles: int = 10,
                   num_repeat_sampler: int = 1000,
                   epsilon: float = 0.01,
                   manipulators_ratio: float = 1.0,
                   num_manipulations_ratio: float = 1.0,
                   ) -> list[pd.DataFrame]:
    """
        Runs an experiment with the given parameters (with already available school capacities) as follows.

        Generates num_repeats_profiles of random profiles of cardinal utility of students.
        For each profile, calls run_experiment_k for k = num_schools (full list) for the Boston mechanism and
        for all k in [1, num_schools] for the limited deferred acceptance mechanism.

        Manipulators and fair students are assigned randomly based on manipulators_ratio.

        In test_system.py, the function is called from run_batch_experiment().

        Parameters:
            num_students (int): Total number of students
            num_schools (int): Total number of available schools
            capacities (np.ndarray): school capacities (shape: (num_schools,))
            capacities_generated (bool): Whether capacities were generated randomly
            num_capacities (int): Number of capacity variants for this experiment
            num_repeats_profiles (int): Number of random preference profiles to simulate (default: 10)
            num_repeat_sampler (int): Number of repetitions of the algorithm on generated preference lists (default: 1000)
            epsilon (float): Minimum utility improvement threshold (default: 0.01)
            manipulators_ratio (float): Fraction of students allowed to manipulate (default: 1.0)
            num_manipulations_ratio: Proportion of allowed manipulations from the num_schools parameter
                                               (or will be set to default: 1.0)

        Returns:
            list[pd.DataFrame]: List of DataFrames, each containing results for one experiment configuration.
                Each DataFrame has columns:
                    - num_students, num_schools, average_runtime, capacities, capacities_generated, num_capacities,
                      num_repeats_profiles, num_repeat_sampler, epsilon, manipulators_ratio, default_fair_num_student,
                      num_manipulations, algorithm, k, k_to_schools_ratio, probabilities, utilities, average_utility,
                      average_number_manipulations, possible_percentage_manipulators,
                      average_actual_percentage_manipulators, average_percentage_unassigned_students,
                      average_percentage_unassigned_fair_students, average_percentage_unassigned_manipulator_students,
                      average_utility_fair_students, average_utility_manipulator_students

        Raises:
            ValueError: If num_students < num_schools.
    """
    if num_students < num_schools:
        raise ValueError("Number of students cannot be less than the number of schools.")

    num_fair = round(num_students * (1 - manipulators_ratio))
    fair_indices = np.random.choice(num_students, num_fair, replace=False)
    num_manipulations = round(num_manipulations_ratio * num_schools)
    k_boston = num_schools  # Only full list for the Boston mechanism are supported now

    experiment_results = []
    start_time = time.time()

    for _ in range(num_repeats_profiles):
        profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)

        # boston algorithm
        probabilities, utilities, manipulators, unassigned_statistic = run_experiment_k(
            algorithm=AlgorithmEnum.BOSTON_MECHANISM,
            num_students=num_students,
            num_schools=num_schools,
            profiles=profiles,
            capacities=capacities,
            num_repeat_sampler=num_repeat_sampler,
            k=k_boston,
            epsilon=epsilon,
            fair_indices=fair_indices,
            num_manipulations=num_manipulations)

        (avg_unassigned_total,
         avg_unassigned_fair,
         avg_unassigned_manipulator,
         avg_utility_fair,
         avg_utility_manipulator) = generate_unassigned_statistic(num_students=num_students,
                                                                  fair_indices=fair_indices,
                                                                  unassigned_statistic=unassigned_statistic,
                                                                  utilities=utilities)

        new_row_boston = make_result_row_run_experiment(
            num_students, num_schools, start_time, capacities, capacities_generated, num_capacities,
            num_repeats_profiles, num_repeat_sampler, epsilon, manipulators_ratio, num_manipulations_ratio,
            num_manipulations, "boston", k_boston, probabilities, utilities, manipulators,
            avg_unassigned_total, avg_unassigned_fair, avg_unassigned_manipulator,
            avg_utility_fair, avg_utility_manipulator
        )

        boston_df = pd.DataFrame(new_row_boston, index=[0])
        experiment_results.append(boston_df)

        # gs algorithm
        for k_gs in range(1, num_schools + 1):
            probabilities, utilities, manipulators, unassigned_statistic = run_experiment_k(
                algorithm=AlgorithmEnum.K_GS_MECHANISM,
                num_students=num_students,
                num_schools=num_schools,
                profiles=profiles,
                capacities=capacities,
                num_repeat_sampler=num_repeat_sampler,
                k=k_gs,
                epsilon=epsilon,
                fair_indices=fair_indices,
                num_manipulations=num_manipulations)

            (avg_unassigned_total,
             avg_unassigned_fair,
             avg_unassigned_manipulator,
             avg_utility_fair,
             avg_utility_manipulator) = generate_unassigned_statistic(num_students=num_students,
                                                                      fair_indices=fair_indices,
                                                                      unassigned_statistic=unassigned_statistic,
                                                                      utilities=utilities)

            new_row_gs = make_result_row_run_experiment(
                num_students, num_schools, start_time, capacities, capacities_generated, num_capacities,
                num_repeats_profiles, num_repeat_sampler, epsilon, manipulators_ratio, num_manipulations_ratio,
                num_manipulations, "gs", k_gs, probabilities, utilities, manipulators,
                avg_unassigned_total, avg_unassigned_fair, avg_unassigned_manipulator,
                avg_utility_fair, avg_utility_manipulator
            )

            gs_df = pd.DataFrame(new_row_gs, index=[0])
            experiment_results.append(gs_df)

    return experiment_results


def run_batch_experiment(tests: list[tuple[int, dict[str, Any]]]) -> pd.DataFrame:
    """
        Runs a batch of school choice experiments and aggregates all results into a single DataFrame.

        For each test configuration, this function launches an experiment (possibly with multiple capacity scenarios),
        collects all result DataFrames, and merges them into a single table. The function
        supports both predefined and randomly generated school capacities, and automatically groups and saves results.

        In test_system.py, the function is called from parallel_run().

        Parameters:
            tests (list[tuple[int, dict[str, Any]]]):
                List of test configurations, where each element is a tuple (test_number, params_dict).
                Each params_dict must contain at least "num_students" and "num_schools" keys, and may include:
                    - capacities: School capacities (shape: (num_schools,))
                    - num_repeats_profiles: Number of random preference profiles to simulate
                                            (or will be set to default: 10)
                    - num_repeat_sampler: Number of repetitions of the algorithm on generated preference lists
                                          (or will be set to default: 1000)
                    - epsilon: Minimum utility improvement threshold (or will be set to default: 0.01)
                    - manipulators_ratio: Fraction of students allowed to manipulate (or will be set to default: 1.0)
                    - num_manipulations_ratio: Fraction of allowed manipulations from the num_schools parameter
                                               (or will be set to default: 1.0)
                    - num_capacities: Number of capacity variants to generate
                                      (if capacities not provided, or will be set to default: 1)

        Returns:
            pd.DataFrame:
                Aggregated experiment results with columns including:
                    - experiment_number, num_students, num_schools, average_runtime, capacities, capacities_generated,
                      num_capacities, num_repeats_profiles, num_repeat_sampler, epsilon, manipulators_ratio,
                      default_fair_num_student, num_manipulations_ratio, num_manipulations, algorithm, k, k_to_schools_ratio, probabilities,
                      utilities, average_utility, average_number_manipulations, possible_percentage_manipulators,
                      average_actual_percentage_manipulators, average_percentage_unassigned_students,
                      average_percentage_unassigned_fair_students, average_percentage_unassigned_manipulator_students,
                      average_utility_fair_students, average_utility_manipulator_students

        Implementation Notes:
            - Uses helper functions `_process_existing_capacities`, `_process_generated_capacities`, and `_execute_experiments` for modularity.
            - The CSV is saved to `./data_out/technical/test_results_{test_number}.csv` where `test_number` is taken from the first test.
              This is done for further correct processing in case of data loss during paralleling in the parallel_run() function.
            - All warnings of type FutureWarning are suppressed during DataFrame concatenation to avoid verbose output.

        Raises:
            KeyError: If required keys ("num_students", "num_schools") are missing in the test configuration.
    """
    # Initialize results DataFrame with predefined columns
    test_results = pd.DataFrame(columns=[
        "num_students", "num_schools", "average_runtime", "capacities",
        "capacities_generated", "num_capacities", "num_repeats_profiles",
        "num_repeat_sampler", "epsilon", "manipulators_ratio",
        "default_fair_num_student", "num_manipulations_ratio", "num_manipulations", "algorithm", "k",
        "probabilities", "utilities", "average_utility",
        "average_utility_fair_students", "average_utility_manipulator_students",
        "possible_percentage_manipulators", "average_actual_percentage_manipulators",
        "average_number_manipulations", "average_percentage_unassigned_students",
        "average_percentage_unassigned_fair_students",
        "average_percentage_unassigned_manipulator_students"
    ])

    # Process each test in batch
    for exp_number, test in tests:
        params = {
            "num_students": test["num_students"],
            "num_schools": test["num_schools"]
        }

        # Add optional parameters
        optional_params = [
            "num_repeats_profiles", "num_repeat_sampler", "epsilon",
            "manipulators_ratio", "num_manipulations_ratio"
        ]
        params.update({key: test[key] for key in optional_params if key in test})

        # Handle capacity generation
        if "capacities" in test:
            test_results = _process_existing_capacities(test, params, exp_number, test_results)
        else:
            test_results = _process_generated_capacities(test, params, exp_number, test_results)

    # Post-processing
    grouped_results = group_test_results(test_results)

    grouped_results.to_csv(
        path_or_buf=f"./data_out/technical/test_results_100_{tests[0][0]}.csv",
        index=False
    )

    return grouped_results


def _process_existing_capacities(
        test: dict[str, Any],
        params: dict[str, Any],
        exp_number: int,
        test_results: pd.DataFrame
) -> pd.DataFrame:
    """Process tests with predefined capacities."""
    params.update({
        "capacities": test["capacities"],
        "capacities_generated": False,
        "num_capacities": 1
    })

    return _execute_experiments(params, exp_number, test_results)


def _process_generated_capacities(
        test: dict[str, Any],
        params: dict[str, Any],
        exp_number: int,
        test_results: pd.DataFrame
) -> pd.DataFrame:
    """Process tests with generating num_capacities capacities."""
    params["capacities_generated"] = True
    num_capacities = test.get("num_capacities", 1)
    params["num_capacities"] = num_capacities

    for _ in range(num_capacities):
        params["capacities"] = generate_school_capacities(
            num_students=test["num_students"],
            num_schools=test["num_schools"]
        )

        test_results = _execute_experiments(params, exp_number, test_results)

    return test_results


def _execute_experiments(
        params: dict[str, Any],
        exp_number: int,
        test_results: pd.DataFrame
) -> pd.DataFrame:
    """Execute experiment and append results to dataframe."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        experiment_results = run_experiment(**params)

        for df in experiment_results:
            df.insert(0, "experiment_number", exp_number + 1)

        test_results = pd.concat([test_results, *experiment_results], ignore_index=True)

    return test_results


def parallel_run(
        tests: list[tuple[int, dict[str, Any]]],
        batch_size: int = 1,
        n_jobs: int = 1,
        display_progress: bool = False
) -> pd.DataFrame:
    """
    Runs tests in parallel batches with optional progress display.

    Splits the input tests into batches of specified size and executes them in parallel,
    combining the results into a single pandas DataFrame.

    Parameters:
        tests (list[tuple[int, dict[str, Any]]]): List of test configurations, where each element is a tuple (test_number, params_dict)
        batch_size (int): Number of tests to process per batch (default: 1).
        n_jobs (int): Number of parallel jobs to run (default: 1).
        display_progress (bool): Whether to display a progress bar (default: False).

    Returns:
        pd.DataFrame: Concatenated results from all test batches.
    """
    # Calculate the number of batches, rounding up
    num_batches = (len(tests) + batch_size - 1) // batch_size

    # Split tests into batches
    batches = [
        tests[i * batch_size: (i + 1) * batch_size]
        for i in range(num_batches)
    ]

    # Configure parallel executor
    executor = Parallel(
        n_jobs=n_jobs,
        backend="loky"
    )

    # Prepare delayed tasks for execution
    tasks = (
        delayed(run_batch_experiment)(tests=batch)
        for batch in batches
    )

    # Wrap tasks with progress bar if requested
    if display_progress:
        tasks = tqdm(tasks, total=num_batches, desc="Processing batches")

    # Execute tasks in parallel
    results = executor(tasks)

    # Concatenate results into a single DataFrame
    experiment_results = pd.concat(results, ignore_index=True)

    experiment_results['utility_rating_in_experiment'] = experiment_results.groupby('experiment_number')[
        'average_utility'].rank(ascending=False, method='dense')

    experiment_results = experiment_results[
        ["experiment_number"] + [col for col in experiment_results.columns if col != "experiment_number"]]

    # Group results by params
    # experiment_results_grouped = group_test_results(experiment_results)

    return experiment_results


def run_full_tests(params_lists: dict[str, list[Any]],
                   batch_size: int = 1,
                   n_jobs: int = 1,
                   display_progress: bool = False,
                   save_path: str = SAVE_PATH_DEFAULT,  # You can change this constant at the beginning of this file
                   print_n_best_results: bool = True,
                   ) -> pd.DataFrame:
    """
    Run full tests with the given parameter lists and return the results.

    Parameters:
        params_lists (dict): Dictionary of parameter lists for the tests.
        batch_size (int): Number of tests to process per batch (default: 1).
        n_jobs (int): Number of parallel jobs to run (default: 1).
        display_progress (bool): Whether to display a progress bar (default: False).

    Returns:
        pd.DataFrame: DataFrame containing the results of the tests.
    """
    tests = generate_tests_from_lists(**params_lists)
    tests = [(i, test) for i, test in enumerate(tests)]

    experiment_results = parallel_run(tests, batch_size, n_jobs, display_progress)

    experiment_results.to_csv(path_or_buf=save_path, index=False)

    if print_n_best_results:
        result, best_k_to_schools_ratio_mean = get_n_best_results(file_path=save_path, n=1)
        print("best k to schools ratios for all experiments:")
        print(result)
        print("average best k to schools ratios for all experiments:")
        print(best_k_to_schools_ratio_mean)

    return experiment_results
