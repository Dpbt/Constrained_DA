import random

import numpy as np

from utils import AlgorithmEnum
from .boston import k_boston_algorithm
from .chinese_parallel import chinese_parallel_mechanism
from .gale_shapley import k_gs_algorithm


def algorithm_sampler(
        algorithm: AlgorithmEnum,
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
        num_repeat: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs multiple simulations of the selected mechanism for the preference lists submitted by students.
    Calculates the probabilities of assigning each student to each school,
    and the probability of each student not being assigned to a school.

    Note: the current implementation only supports equal prioritization of schools with respect to students

    Parameters:
        algorithm (AlgorithmEnum): Matching mechanism to test (BOSTON_MECHANISM/K_GS_MECHANISM/CHINESE_PARALLEL_MECHANISM)
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        preferences (np.ndarray): Student preference matrix (shape: (num_students, k) for BOSTON_MECHANISM/K_GS_MECHANISM
                                                    and shape: (num_students, num_schools) for CHINESE_PARALLEL_MECHANISM)
                                  Each row represents a student's school preferences per round
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        k (int): Length of preference lists (k <= num_schools)
        num_repeat (int): Number of iterations of algorithm repetition to average the results

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - Assignment probabilities (shape: (num_students, num_schools + 1))
        - Unassignment probabilities (shape: (num_students,))

    Raises:
        ValueError: If unknown algorithm type provided

    Implementation Notes:
        1. Randomization Process:
            - Generates unique school preference permutations
        2. Statistics Collection:
            - Tracks school assignments in first num_schools columns
            - Records unassignments in last column
        3. Probability Calculation:
            - Normalizes counts by num_repeat
            - Maintains separate unassignment statistics
    """
    statistic = np.zeros((num_students, num_schools + 1))
    student_rank_default = tuple(range(num_students))

    def generate_random_permutations(elements):
        """Infinite generator of random permutations using unbiased shuffle"""
        while True:
            yield random.sample(elements, len(elements))

    student_ranks_iterator = generate_random_permutations(student_rank_default)

    # Algorithm selection
    if algorithm == AlgorithmEnum.BOSTON_MECHANISM:
        algorithm_func = k_boston_algorithm
    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        algorithm_func = k_gs_algorithm
    elif algorithm == AlgorithmEnum.CHINESE_PARALLEL_MECHANISM:
        algorithm_func = chinese_parallel_mechanism
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}. "
                         f"Only AlgorithmEnum.K_GS_MECHANISM, "
                         f"AlgorithmEnum.BOSTON_MECHANISM and"
                         f"AlgorithmEnum.CHINESE_PARALLEL_MECHANISM are supported now.")

    # Simulation loop
    for _ in range(num_repeat):
        # current_school_preferences = next(student_ranks_iterator)
        current_school_preferences = {student: rank for rank, student in enumerate(next(student_ranks_iterator))}

        assignments, unassigned_students = algorithm_func(
            num_students=num_students,
            num_schools=num_schools,
            preferences=preferences,
            capacities=capacities,
            k=k,
            school_preferences=current_school_preferences
        )

        # Update statistics
        for school, students in assignments.items():
            for student in students:
                statistic[student, school] += 1

        for student in unassigned_students:
            statistic[student, num_schools] += 1

    # Normalize results
    probabilities = statistic / num_repeat
    unassigned_statistic = statistic[:, num_schools] / num_repeat

    return probabilities, unassigned_statistic