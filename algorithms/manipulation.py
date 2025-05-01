import random

import numpy as np

from utils import (AlgorithmEnum, generate_k_restricted_preferences, generate_possible_manipulations,
                   calculate_utilities_from_probs_individual)
from .probs_estimator import k_gs_algorithm_prob_individual


def _get_manipulation_candidates(
        manipulators: np.ndarray,
        fair_indices: np.ndarray,
        max_manipulations: int
) -> list[int]:
    """Identifies eligible students for manipulation."""
    return [
        i for i in range(len(manipulators))
        if manipulators[i] < max_manipulations and i not in fair_indices
    ]


def _find_optimal_manipulation(
        student: int,
        preferences: np.ndarray,
        prob_func: callable,
        profiles: np.ndarray,
        capacities: np.ndarray,
        k: int,
        num_schools: int,
        epsilon: float,
        current_utility: float
) -> tuple[np.ndarray, float]:
    """Searches for utility-improving preference modifications."""
    best_manipulation_gain = 0
    best_manipulation = preferences[student]

    for new_preference in generate_possible_manipulations(num_schools, preferences[student], k):
        modified_prefs = preferences.copy()
        modified_prefs[student] = new_preference

        new_probs = prob_func(num_schools=num_schools,
                              preferences=modified_prefs,
                              capacities=capacities,
                              k=k,
                              student=student)
        new_utility = calculate_utilities_from_probs_individual(student=student,
                                                                probabilities=new_probs,
                                                                profiles=profiles)
        utility_gain = new_utility - current_utility

        if utility_gain > best_manipulation_gain and utility_gain > epsilon:
            best_manipulation_gain = utility_gain
            best_manipulation = new_preference

    return best_manipulation, best_manipulation_gain


def manipulation_algorithm(
        algorithm: AlgorithmEnum,
        num_students: int,
        num_schools: int,
        profiles: np.ndarray,
        capacities: np.ndarray,
        k: int,
        epsilon: float,
        fair_indices: np.ndarray,
        num_manipulations: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements student manipulation of lists based on their estimation of the probability and
    expected utility for each school on the list.

    Parameters:
        algorithm (AlgorithmEnum): Target mechanism for manipulation analysis (Only K_GS_MECHANISM avalaible
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        profiles (np.ndarray): Matrix of cardinal utility of students from schools (shape: (num_students, num_schools))
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        k (int): Length of preference lists (k <= num_schools)
        epsilon (float): Minimum utility improvement threshold
        fair_indices (np.ndarray): Indices of non-manipulating students
        num_manipulations (int): Max manipulations per student

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - Modified preference lists
        - Manipulation counters per student

    Implementation Notes:
        1. Iterative Improvement: Greedily optimizes student utilities
        2. Fairness Constraints: Protects fair_indices from manipulation
        3. Termination Conditions:
            - Max manipulations reached
            - No improving manipulations found
        4. Search Strategy: Randomized student processing
    """
    # Initialize algorithm components
    if algorithm == AlgorithmEnum.K_GS_MECHANISM:
        prob_func = k_gs_algorithm_prob_individual
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}. "
                         f"Only AlgorithmEnum.K_GS_MECHANISM is supported now.")

    # Initialize data structures
    preferences = generate_k_restricted_preferences(profiles, k)
    manipulators = np.zeros(num_students)
    max_total_manipulations = num_manipulations * (num_students - len(fair_indices))
    manipulation_active = True

    # Main optimization loop
    while np.sum(manipulators) < max_total_manipulations and manipulation_active:
        manipulation_active = False
        candidates = _get_manipulation_candidates(manipulators, fair_indices, num_manipulations)

        # Process candidates in randomized order
        random.shuffle(candidates)

        for student in candidates:
            current_probs = prob_func(num_schools=num_schools,
                                      preferences=preferences,
                                      capacities=capacities,
                                      k=k,
                                      student=student)
            current_utility = calculate_utilities_from_probs_individual(student=student,
                                                                        probabilities=current_probs,
                                                                        profiles=profiles)

            # Search for optimal manipulation
            best_manipulation, best_manipulation_gain = _find_optimal_manipulation(
                student=student,
                preferences=preferences,
                prob_func=prob_func,
                profiles=profiles,
                capacities=capacities,
                k=k,
                num_schools=num_schools,
                epsilon=epsilon,
                current_utility=current_utility
            )

            # Apply successful manipulation
            if best_manipulation_gain > 0:
                preferences[student] = best_manipulation
                manipulators[student] += 1
                manipulation_active = True
                break

    return preferences, manipulators