import numpy as np
import pandas as pd
import itertools
import random


random.seed(42)
np.random.seed(42)


def generate_random_profiles(num_students: int, num_schools: int) -> np.ndarray:
    """
    Generates (same ordinal but different cardinal) normalized random preference profiles for students over schools.

    Preferences are generated using uniform random distribution, normalized to sum
    to 1 for each student, and sorted in descending order.

    Parameters:
        num_students (int): Number of students to generate preferences for
        num_schools (int): Number of schools to generate preferences for

    Returns:
        numpy.ndarray: 2D array of shape (num_students, num_schools) where:
        - Each row represents a student's preferences
        - Preferences are normalized (sum to 1)
        - Preferences are sorted in descending order per student

    Example:
        > generate_random_profiles(2, 2)
        np.array([[0.634, 0.366],
               [0.781, 0.219]])
    """
    profiles = np.random.rand(num_students, num_schools)
    profiles = profiles / profiles.sum(axis=1, keepdims=True)
    profiles = np.sort(profiles, axis=1)[:, ::-1]
    return profiles


def generate_school_capacities(num_students: int, num_schools: int) -> np.ndarray:
    """
    Generates random school capacities that sum to total number of students.

    Creates non-zero capacities through random partitioning of total student count.
    Uses sorted random points to split student pool between schools.

    Parameters:
        num_students (int): Total number of students to distribute
        num_schools (int): Number of schools requiring capacities

    Returns:
        np.ndarray: 1D array of shape (num_schools,) containing:
        - Strictly positive integers (>=1)
        - Elements sum to num_students
        - Random distribution between schools

    Example:
        > generate_school_capacities(100, 4)
        np.array([24, 31, 22, 23])  # Sum=100
    """
    capacities = np.random.choice(np.arange(1, num_students), size=num_schools-1, replace=False)
    capacities = np.sort(np.concatenate(([0], capacities, [num_students])))
    capacities = np.diff(capacities)
    return capacities


def generate_k_restricted_preferences(profiles: np.ndarray, k: int) -> np.ndarray:
    """
       Generates top-k school preferences from probability distribution profiles.

       Converts preference probabilities to school rankings and selects top-k choices.

       Parameters:
           profiles (np.ndarray): Matrix of cardinal utility of students from schools
                                 Shape: (num_students, num_schools)
           k (int): Number of top preferences to select (1 ≤ k ≤ num_schools)

       Returns:
           np.ndarray: Integer array of shape (num_students, k) containing
                      school indices (0-based) sorted by preference strength

       Example:
           > profiles = generate_random_profiles(4, 3)
           > generate_k_restricted_preferences(profiles, 2)
           np.array([[0 1]
                    [0 1]
                    [0 1]
                    [0 1]])
           # In this case, all students have schools 0 and 1, since generate_random_profiles
           # returns the same ordinal preferences
    """
    preferences = np.argsort(profiles, axis=1)[:, -1 : -k - 1 : -1]
    return preferences


def calculate_utilities(
    num_students: int,
    assignments: dict[int, list[int]],
    profiles: np.ndarray
) -> dict[int, float]:
    """
    Calculates individual utility for students based on their school assignments.

    Parameters:
        num_students (int): Total number of students (for result initialization)
        assignments (dict[int, list[int]]): School assignments {school_id: [student_ids]}
        profiles (np.ndarray): Preference profiles of students
                              Shape: (num_students, num_schools), dtype: float

    Returns:
        dict[int, float]: Mapping {student_id: utility_score} where:
        - utility_score ∈ [0, 1]
        - Score represents preference strength for assigned school

    Example:
        > profiles = np.array([
        ...     [0.9, 0.1],  # Student 0
        ...     [0.8, 0.2],  # Student 1
        ...     [0.7, 0.3]   # Student 2
        ... ])
        > assignments = {
        ...     0: [1, 2],  # School 0 gets students 1 and 2
        ...     1: [0]      # School 1 gets student 0
        ... }
        > calculate_utilities(3, assignments, profiles)
        {0: 0.1, 1: 0.8, 2: 0.7}
    """
    student_utility = {i: 0.0 for i in range(num_students)}

    for school_id, students in assignments.items():
        for student in students:
            student_utility[student] = float(profiles[student, school_id])

    return student_utility


def calculate_utilities_from_probs(
    num_students: int,
    num_schools: int,
    probabilities: np.ndarray,
    profiles: np.ndarray
) -> np.ndarray:
    """
    Calculation of expected student utilities using school assignment probabilities.

    Parameters:
        num_students (int): Total number of students (consistency check)
        num_schools (int): Number of schools to consider (<= matrix columns)
        probabilities (NDArray[float64]): Assignment probability matrix
            Shape: (num_students, num_schools+), each row sums to 1
        profiles (NDArray[float64]): Preference strength matrix
            Shape: (num_students, num_schools+), values normalized [0,1]

    Returns:
        NDArray[float64]: 1D array of expected utilities (shape: num_students)

    Example:
        > profiles = np.array([
        ...     [0.9, 0.1, 0.0],  # Student 0 preferences
        ...     [0.6, 0.3, 0.1]   # Student 1 preferences
        ... ], dtype=np.float64)
        > probabilities = np.array([
        ...     [0.8, 0.2, 0.0],  # Student 0 assignment probs
        ...     [0.1, 0.7, 0.2]   # Student 1 assignment probs
        ... ], dtype=np.float64)
        > calculate_utilities_from_probs(2, 3, probabilities, profiles)
        array([0.74 0.29])  # 0.9*0.8 + 0.1*0.2 = 0.74 | 0.6*0.1 + 0.3*0.7 + 0.1*0.2= 0.29
    """
    return np.sum(probabilities[:, :num_schools] * profiles[:, :num_schools], axis=1)


def calculate_utilities_from_probs_individual(student: int, num_schools: int, probabilities: np.ndarray, profiles: np.ndarray):
    student_utility = np.sum(probabilities * profiles[student])
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


def generate_tests_from_lists(**param_lists):
    param_names = list(param_lists.keys())
    combinations = itertools.product(*param_lists.values())

    result = []
    for combo in combinations:
        dict_item = {}
        for name, value in zip(param_names, combo):
            if name == 'num_manipulations':
                dict_item[name] = round(value * dict_item.get('num_schools', 1))
            else:
                dict_item[name] = value
        result.append(dict_item)

    return result


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
    # print(calculate_utilities_from_probs(3, 3, prob, profiles))
    # print(calculate_utilities_from_prob_v2(3, 3, prob, profiles))

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

    print(generate_possible_manipulations(num_schools=4, preferences=np.array([0, 1]), k=2))
