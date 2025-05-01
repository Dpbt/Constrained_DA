import numpy as np


def generate_statistic(num_schools: int, preferences: np.ndarray, k: int) -> np.ndarray:
    """
    Generates a positional frequency matrix of school preferences across all students.

    Parameters:
        num_schools (int): Total number of available schools
        preferences (np.ndarray): 2D array of student preferences
                                 Shape: (num_students, k), dtype: integer
        k (int): Length of preference lists (k <= num_schools)

    Returns:
        np.ndarray: Frequency matrix of shape (k, num_schools) where:

        - matrix[i, j] = number of times school j appears in position i+1

        - Rows represent preference positions (1st, 2nd,...kth choice)

        - Columns represent schools

    Example:
        > preferences = np.array([[0, 1], [1, 0], [0, 2]], dtype=int)

        > generate_statistic(3, preferences, 2)

        np.array([[2., 1., 0.],  # Position 1 counts: school0(2), school1(1) \n
                  [1., 1., 1.]]) # Position 2 counts: school0(1), school1(1), school2(1)
    """
    statistic = np.zeros((k, num_schools))

    for preference in preferences:
        for school_ind in range(k):
            statistic[school_ind, preference[school_ind]] += 1

    return statistic


def generate_unassigned_statistic(
        num_students: int,
        fair_indices: np.ndarray,
        unassigned_statistic: np.ndarray,
        utilities: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Calculate statistics for assigned and unassigned students, comparing fair students and manipulators.

    Parameters:
        num_students (int): Total number of students
        fair_indices (np.ndarray): Indices of non-manipulating students (shape: (n,))
        unassigned_statistic (np.ndarray): An array indicating the probability of each student not getting into any school
                                          Shape: (num_students,)
        utilities (np.ndarray): Utility values for all students (shape: (num_students,))

    Returns:
        tuple[float, float, float, float, float]: Five metrics:

        1. Average % of unassigned students (all)
        2. Average % of unassigned fair students
        3. Average % of unassigned manipulator students
        4. Average utility of fair students
        5. Average utility of manipulator students

    Example:
        > num_students = 4

        > fair_indices = np.array([0, 1])

        > unassigned = np.array([1, 0, 1, 0])  # Students 0/2 unassigned

        > utilities = np.array([0.0, 0.8, 0.0, 0.6])

        > generate_unassigned_statistic(num_students, fair_indices, unassigned, utilities)

        (50.0, 50.0, 50.0, 0.4, 0.3)  # 50% overall, 50% fair, 50% manipulators, utilities 0.4 for fair and 0.3 for manipulators
    """
    fair_mask = np.zeros(num_students, dtype=bool)
    fair_mask[fair_indices] = True

    # Split utilities
    fair_utilities = utilities[fair_mask]
    manipulator_utilities = utilities[~fair_mask]

    # Split assignment status
    unassigned_fair = unassigned_statistic[fair_mask]
    unassigned_manipulator = unassigned_statistic[~fair_mask]

    # Count populations
    num_fair = len(fair_indices)
    num_manipulators = num_students - num_fair

    # Calculate percentages
    avg_unassigned_total = (
        (np.sum(unassigned_statistic) / num_students * 100) if num_students > 0 else 0
    )
    avg_unassigned_fair = (np.mean(unassigned_fair) * 100) if num_fair > 0 else 0
    avg_unassigned_manipulator = (
        (np.mean(unassigned_manipulator) * 100) if num_manipulators > 0 else 0
    )

    # Calculate utilities
    avg_utility_fair = np.mean(fair_utilities) if num_fair > 0 else 0
    avg_utility_manipulator = (
        np.mean(manipulator_utilities) if num_manipulators > 0 else 0
    )

    return (
        avg_unassigned_total,
        avg_unassigned_fair,
        avg_unassigned_manipulator,
        avg_utility_fair,
        avg_utility_manipulator,
    )

