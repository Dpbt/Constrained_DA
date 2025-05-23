import numpy as np
import itertools
from collections import defaultdict
from algorithms import k_gs_algorithm, chinese_parallel_mechanism
from utils import (AlgorithmEnum, generate_possible_preferences_chinese, generate_possible_preferences_k_gs,
                   generate_symmetric_preferences)
from analysis import find_nash_equilibrium


def all_preferences_test(
        num_students: int,
        num_schools: int,
        k: int,
        capacities: np.ndarray,
        profiles: np.ndarray,
        algorithm: AlgorithmEnum = AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Exhaustively tests all symmetric preference combinations against school permutations,
    as well as individual deviations from symmetric strategies.

    Conducts full factorial analysis of student-school matching by evaluating:

    - All possible symmetric student preference profiles and individual deviations from them

    - Every school preference permutation

    - Specified matching algorithm behavior

    Parameters:
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        k (int): Block or list constraint size (mechanism dependent)
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        profiles (np.ndarray): Matrix of utility profiles (shape: (num_students, num_schools))
        algorithm (AlgorithmEnum): Matching mechanism to test. Available options: AlgorithmEnum.CHINESE_PARALLEL_MECHANISM, AlgorithmEnum.K_GS_MECHANISM

    Returns:
        list[tuple[np.ndarray, np.ndarray]]:
        - preferences: Generated symmetric preference matrix
        - avg_utilities: Average utilities across all school permutations

    Raises:
    ValueError: If unknown algorithm type provided (supported: AlgorithmEnum.CHINESE_PARALLEL_MECHANISM, AlgorithmEnum.K_GS_MECHANISM)

    Implementation Workflow:
        1. Algorithm Setup:

           - Selects appropriate preference generator

           - Chooses matching algorithm implementation
        2. Preference Generation:

           - Creates symmetric student preference profiles

           - Generates all school preference permutations
        3. Matching Simulation:

           - Runs algorithm for each (student_pref, school_pref) pair

           - Tracks assignment utilities
        4. Result Aggregation:

           - Calculates average utilities per preference profile

           - Compiles comprehensive test report
    """
    # Algorithm configuration
    if algorithm == AlgorithmEnum.CHINESE_PARALLEL_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_chinese
        algorithm_func = chinese_parallel_mechanism
    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_k_gs
        algorithm_func = k_gs_algorithm
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}. "
                         f"Only AlgorithmEnum.K_GS_MECHANISM and "
                         f"AlgorithmEnum.CHINESE_PARALLEL_MECHANISM are supported now.")

    # School preference permutations
    school_preferences_default = [i for i in range(num_students)]
    school_preferences = list(itertools.permutations(school_preferences_default))

    # Generate symmetric student preferences
    all_symmetric_preferences = generate_symmetric_preferences(
        num_students=num_students,
        k=k,
        num_schools=num_schools,
        profiles=profiles,
        get_possible_preferences_func=get_possible_preferences_func
    )

    profile_groups = defaultdict(list)
    for idx, profile in enumerate(profiles):
        profile_groups[tuple(profile)].append(idx)

    all_profiles_set = set()

    for sym_profile in all_symmetric_preferences:
        all_profiles_set.add(tuple(map(tuple, sym_profile)))

        for group in profile_groups.values():
            representative = group[0]
            for alt_pref in get_possible_preferences_func(num_schools, k):
                if np.array_equal(alt_pref, sym_profile[representative]):
                    continue
                new_profile = sym_profile.copy()
                new_profile[representative] = alt_pref
                all_profiles_set.add(tuple(map(tuple, new_profile)))

    all_profiles = [np.array(profile, dtype=np.int32) for profile in all_profiles_set]

    results = []
    for i, preferences in enumerate(all_profiles):
        preferences_utils = np.zeros(num_students)

        # Test against all school preference orders
        for school_preference in school_preferences:
            school_preference_dict = {student: rank for rank, student in enumerate(school_preference)}
            school_assignments, unassigned = algorithm_func(
                num_students=num_students,
                num_schools=num_schools,
                preferences=preferences,
                capacities=capacities,
                k=k,
                school_preferences=school_preference_dict,
            )

            # Calculate individual utilities
            student_assignments = {}
            for school, students in school_assignments.items():
                for student in students:
                    student_assignments[student] = school

            curr_utils = np.array([
                (profiles[student][student_assignments[student]]
                 if student not in unassigned else 0)
                for student in range(num_students)
            ])
            preferences_utils += curr_utils

        # Average across all school preferences
        preferences_utils /= len(school_preferences)
        results.append((preferences, preferences_utils))

    return results


def run_all_preferences_test():
    """
    Run all preferences test with given parameters and return results.
    """
    # Some examples of profiles from the article

    profiles = np.array([
        [54, 23, 15, 8],
        [54, 23, 15, 8],
        [38, 32, 30, 0],
        [38, 32, 30, 0],
    ])

    # profiles = np.array([
    #     [90, 6, 4, 0],
    #     [90, 8, 2, 0],
    #     [90, 9, 1, 0],
    #     [90, 10, 0, 0],
    # ])

    # profiles = np.array([
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 4, 1],
    # ])

    num_schools = profiles.shape[1]
    num_students = profiles.shape[0]
    k = 2
    capacities = np.array([1 for _ in range(num_schools)])

    results = all_preferences_test(
        num_schools=num_schools,
        num_students=num_students,
        k=k,
        capacities=capacities,
        profiles=profiles,
        algorithm=AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
        # algorithm=AlgorithmEnum.K_GS_MECHANISM,
    )

    find_nash_equilibrium(results=results, profiles=profiles, symmetric=True)


if __name__ == "__main__":
    run_all_preferences_test()
