import itertools
import time

import numpy as np
import pandas as pd


def generate_tests_from_lists(**param_lists) -> list[dict]:
    """
    Generates parameter combinations for experiments from input lists.
    Handles num_manipulations in a special way, converting the proportion into a direct number
    of allowed manipulations as round(value * num_schools)

    Parameters:
        **param_lists: Keyword arguments with parameter lists. Example:
            num_students=[100, 200],
            num_schools=[5, 10],
            num_manipulations=[0.1, 0.2]  # Will be multiplied by num_schools and rounded then

    Returns:
        list[dict]: List of parameter dictionaries with all possible combinations

    Example:
        > generate_tests_from_lists(
             num_students=[100, 200], \n
             num_schools=[5], \n
             num_manipulations=[0.2]
          )

        [{'num_students': 100, 'num_schools': 5, 'num_manipulations': 1},
         {'num_students': 200, 'num_schools': 5, 'num_manipulations': 1}]

    Special Cases:
        - For 'num_manipulations': rounds (value * num_schools)
        - Maintains original values for other parameters
        - Empty input returns empty list
    """
    param_names = list(param_lists.keys())
    combinations = itertools.product(*param_lists.values())

    result = []
    for combo in combinations:
        dict_item = {}
        for name, value in zip(param_names, combo):
            dict_item[name] = value
        result.append(dict_item)

    return result


def group_test_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates experiment results by test parameters and calculates average metrics.

    Processing Logic:
        1. Groups by configuration parameters
        2. Calculates mean values for performance metrics
        3. Preserves original experiment numbers
        4. Sorts results for analysis

    Parameters:
        df (pd.DataFrame): Raw experiment results containing:
            - Configuration parameters (num_students, num_schools, etc.)
            - Performance metrics (runtime, utilities, etc.)
            - Experiment identifiers (experiment_number)

    Returns:
        pd.DataFrame: Aggregated results with:

            - Original configuration parameters

            - Averaged performance metrics

            - Sorted by experiment_number, k, and algorithm
    """
    pd.set_option('display.max_columns', None)
    # Configuration parameters for grouping
    groupby_columns = [
        "num_students",
        "num_schools",
        "capacities_generated",
        "num_capacities",
        "num_repeats_profiles",
        "num_repeat_sampler",
        "epsilon",
        "manipulators_ratio",
        "default_fair_num_student",
        "num_manipulations_ratio",
        "algorithm",
        "k",
        "k_to_schools_ratio",
        "possible_percentage_manipulators",
    ]

    # Metrics for averaging
    average_columns = [
        "num_manipulations",
        "average_runtime",
        "average_utility",
        "average_utility_fair_students",
        "average_utility_manipulator_students",
        "average_actual_percentage_manipulators",
        "average_number_manipulations",
        "average_percentage_unassigned_students",
        "average_percentage_unassigned_fair_students",
        "average_percentage_unassigned_manipulator_students",
    ]

    # print(df)

    # Grouping and aggregation
    grouped_df = (
        df.groupby(groupby_columns)
        .agg(
            experiment_number=("experiment_number", "first"),
            **{col: (col, "mean") for col in average_columns}
        )
        .reset_index()
    )

    # print(grouped_df)

    # Column reordering
    grouped_df = grouped_df[
        ["experiment_number"]
        + [col for col in grouped_df.columns if col != "experiment_number"]
        ]

    # Sorting logic
    grouped_df = grouped_df.sort_values(
        by=["experiment_number", "k", "algorithm"], ascending=[True, True, False]
    )

    return grouped_df


def make_result_row_run_experiment(
        num_students, num_schools, start_time, capacities, capacities_generated, num_capacities,
        num_repeats_profiles, num_repeat_sampler, epsilon, manipulators_ratio, num_manipulations_ratio,
        num_manipulations, algorithm, k, probabilities, utilities, manipulators,
        avg_unassigned_total, avg_unassigned_fair, avg_unassigned_manipulator,
        avg_utility_fair, avg_utility_manipulator
) -> dict[str, list]:
    """
    Forms a row for the dataframe in run_experiment
    """
    num_fair = round(num_students * (1 - manipulators_ratio))
    return {
        "num_students": [num_students],
        "num_schools": [num_schools],
        "average_runtime": [time.time() - start_time],
        "capacities": [capacities],
        "capacities_generated": [capacities_generated],
        "num_capacities": [num_capacities],
        "num_repeats_profiles": [num_repeats_profiles],
        "num_repeat_sampler": [num_repeat_sampler],
        "epsilon": [epsilon],
        "manipulators_ratio": [manipulators_ratio],
        "default_fair_num_student": [num_fair],
        "num_manipulations_ratio": [num_manipulations_ratio],
        "num_manipulations": [num_manipulations],
        "algorithm": [algorithm],
        "k": [k],
        "k_to_schools_ratio": [round(k / num_schools, 3)],
        "probabilities": [probabilities],
        "utilities": [utilities],
        "average_utility": [np.sum(utilities) / num_students],
        "average_number_manipulations": [np.sum(manipulators)],
        "possible_percentage_manipulators": [(num_students - num_fair) / num_students * 100],
        "average_actual_percentage_manipulators": [np.count_nonzero(manipulators) / num_students * 100],
        "average_percentage_unassigned_students": [avg_unassigned_total],
        "average_percentage_unassigned_fair_students": [avg_unassigned_fair],
        "average_percentage_unassigned_manipulator_students": [avg_unassigned_manipulator],
        "average_utility_fair_students": [avg_utility_fair],
        "average_utility_manipulator_students": [avg_utility_manipulator]
    }