import numpy as np
import pandas as pd
from numba import njit, jit

# откуда фиксируется seed ?


def generate_random_profiles(num_students: int, num_schools: int):
    # Генерация случайных предпочтений
    profiles = np.random.rand(num_students, num_schools)
    profiles = profiles / profiles.sum(axis=1, keepdims=True)
    profiles = np.sort(profiles, axis=1)[:, ::-1]
    return profiles


def generate_school_capacities(num_students: int, num_schools: int):
    # мб нужно добавить параметр отвечающий за равномерность распределений
    capacities = np.random.choice(np.arange(1, num_students), size=num_schools-1, replace=False)
    capacities = np.sort(np.concatenate(([0], capacities, [num_students])))
    capacities = np.diff(capacities)
    return capacities


def generate_k_restricted_preferences(profiles: np.ndarray, k: int):
    preferences = np.argsort(profiles, axis=1)[:, -1:-k-1:-1]
    return preferences


def calculate_utility(num_students: int, assignments: dict[int, list[int]], profiles: np.ndarray):
    student_utility = {i: 0 for i in range(num_students)}

    for school_id, students in assignments.items():
        for student in students:
            student_utility[student] = profiles[student][school_id]

    return student_utility


def calculate_utilities_from_prob(num_students: int, num_schools: int, probabilities: np.ndarray, profiles: np.ndarray):
    student_utility = np.zeros(num_students)
    for student in range(num_students):
        student_utility[student] = np.sum(probabilities[student][:num_schools] * profiles[student])
    return student_utility


def generate_subsets(nums, k):
    n = len(nums)
    result = []

    for i in range(1 << n):  # 2^n
        subset = []
        for j in range(n):
            if (i & (1 << j)) > 0:  # Проверка j-го бита
                subset.append(nums[j])
        if len(subset) == k:
            result.append(subset)

    return result


def generate_possible_manipulations(num_schools: int, preferences: np.ndarray, k: int
                                    ) -> np.ndarray:
    if len(preferences) != k or not np.all(np.diff(preferences) > 0):
        raise ValueError("Preferences must be a sorted array of length k.")

    preferences_set = set(preferences)
    manipulations = []

    for i in range(k):
        current_preference = preferences[i]

        for next_school in range(current_preference + 1, num_schools):
            if next_school not in preferences_set:
                new_preferences = preferences.copy()
                new_preferences[i] = next_school
                manipulations.append(np.sort(new_preferences))
                break

        # for prev_school in range(current_preference - 1, -1, -1):
        #     if prev_school not in preferences_set:
        #         new_preferences = preferences.copy()
        #         new_preferences[i] = prev_school
        #         manipulations.append(np.sort(new_preferences))
        #         break  # Прерываем цикл после первой замены

    unique_manipulations = np.unique(manipulations, axis=0)

    return unique_manipulations


def generate_statistic(num_schools: int, preferences: np.ndarray, k: int):
    statistic = np.zeros((k, num_schools))

    for preference in preferences:
        for school_ind in range(k):
            statistic[school_ind, preference[school_ind]] += 1

    return statistic


def generate_unassigned_statistic(num_students: int,
                                  fair_indices: np.ndarray,
                                  unassigned_statistic: np.ndarray,
                                  utilities: np.ndarray):
    fair_mask = np.zeros(num_students, dtype=bool)
    fair_mask[fair_indices] = True

    fair_utilities = utilities[fair_mask]
    manipulator_utilities = utilities[~fair_mask]
    unassigned_fair_statistic = unassigned_statistic[fair_mask]
    unassigned_manipulator_statistic = unassigned_statistic[~fair_mask]

    average_percentage_unassigned_students = (np.sum(unassigned_statistic)) / num_students * 100
    average_percentage_unassigned_fair_students = (np.sum(unassigned_fair_statistic)) / len(fair_indices) * 100
    average_percentage_unassigned_manipulator_students = ((np.sum(unassigned_manipulator_statistic)) /
                                                          (num_students - len(fair_indices)) * 100)
    average_utility_fair_students = (np.sum(fair_utilities)) / len(fair_indices)
    average_utility_manipulator_students = (np.sum(manipulator_utilities)) / (num_students - len(fair_indices))

    return (average_percentage_unassigned_students,
            average_percentage_unassigned_fair_students,
            average_percentage_unassigned_manipulator_students,
            average_utility_fair_students,
            average_utility_manipulator_students)


import numpy as np


def generate_unassigned_statistic(num_students: int,
                                  fair_indices: np.ndarray,
                                  unassigned_statistic: np.ndarray,
                                  utilities: np.ndarray):
    fair_mask = np.zeros(num_students, dtype=bool)
    fair_mask[fair_indices] = True

    fair_utilities = utilities[fair_mask]
    manipulator_utilities = utilities[~fair_mask]
    unassigned_fair_statistic = unassigned_statistic[fair_mask]
    unassigned_manipulator_statistic = unassigned_statistic[~fair_mask]

    num_fair_students = len(fair_indices)
    num_manipulator_students = num_students - num_fair_students

    average_percentage_unassigned_students = (
                np.sum(unassigned_statistic) / num_students * 100) if num_students > 0 else 0

    average_percentage_unassigned_fair_students = (
                np.mean(unassigned_fair_statistic)  * 100) if num_fair_students > 0 else 0

    average_percentage_unassigned_manipulator_students = (np.mean(
        unassigned_manipulator_statistic) * 100) if num_manipulator_students > 0 else 0

    average_utility_fair_students = np.mean(fair_utilities) if num_fair_students > 0 else 0
    average_utility_manipulator_students = np.mean(manipulator_utilities) if num_manipulator_students > 0 else 0

    return (average_percentage_unassigned_students,
            average_percentage_unassigned_fair_students,
            average_percentage_unassigned_manipulator_students,
            average_utility_fair_students,
            average_utility_manipulator_students)


def group_test_results(df: pd.DataFrame) -> pd.DataFrame:
    groupby_columns = ["num_students", "num_schools", "capacities_generated", "num_capacities", "num_repeats_profiles",
                       "num_repeat_sampler", "epsilon", "manipulators_ratio", "default_fair_num_student",
                       "num_manipulations", "algorithm", "k", "possible_percentage_manipulators"]

    average_columns = ["average_runtime", "average_utility", "average_utility_fair_students",
                       "average_utility_manipulator_students", "average_actual_percentage_manipulators",
                       "average_number_manipulations", "average_percentage_unassigned_students",
                       "average_percentage_unassigned_fair_students", "average_percentage_unassigned_manipulator_students"]

    grouped_df = df.groupby(groupby_columns).agg(
        experiment_number=('experiment_number', 'first'),
        **{col: (col, 'mean') for col in average_columns}
    ).reset_index()

    grouped_df = grouped_df[
        ['experiment_number'] + [col for col in grouped_df.columns if col != 'experiment_number']]

    grouped_df = grouped_df.sort_values(
        by=['experiment_number', "k", "algorithm"],
        ascending=[True, True, False]
    )

    return grouped_df


if __name__ == '__main__':
    x = generate_random_profiles(10, 5)

    # num_schools = 5
    # preferences = np.array([1, 2, 4])
    # k = len(preferences)
    # result = generate_possible_manipulations(num_schools, preferences, k)
    # print(result)

    # prob = np.array([[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333],
    #                  [0.33333333, 0.33333333, 0.33333333]])
    # profiles = np.array([[0.4365224, 0.29180088, 0.27167672], [0.55439957, 0.4264105, 0.01918993],
    #                      [0.47070475, 0.39570134, 0.13359391]])
    # print(calculate_utilities_from_prob(3, 3, prob, profiles))

    # df = pd.read_csv('experiment_results.csv')
    #
    # grouped_results = group_test_results(df)
    # print(grouped_results)
    #
    # grouped_results.to_csv('experiment_results_grouped.csv', index=False)

    utilities = np.array([10, 20, 30, 40, 50])

    (average_percentage_unassigned_students,
     average_percentage_unassigned_fair_students,
     average_percentage_unassigned_manipulator_students,
     average_utility_fair_students,
     average_utility_manipulator_students) = generate_unassigned_statistic(num_students = 5,
                                      fair_indices = np.array([1 ,2]),
                                      unassigned_statistic = np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                                      utilities = utilities)

    print(average_percentage_unassigned_students,
     average_percentage_unassigned_fair_students,
     average_percentage_unassigned_manipulator_students,
     average_utility_fair_students,
     average_utility_manipulator_students)



