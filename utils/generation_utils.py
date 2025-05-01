import itertools
from typing import Callable

import numpy as np


def generate_random_profiles(num_students: int, num_schools: int) -> np.ndarray:
    """
    Generates (same ordinal but different cardinal) normalized random preference profiles for students over schools.

    Preferences are generated using uniform random distribution, normalized to sum
    to 1 for each student, and sorted in descending order.

    Parameters:
        num_students (int): Total number of students to generate preferences for
        num_schools (int): Total number of available schools to generate preferences for

    Returns:
        Matrix of cardinal utility of students from schools

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
        num_students (int): Total number of students
        num_schools (int): Total number of available schools requiring capacities

    Returns:
        School capacities

        np.ndarray: 1D array of shape (num_schools,) containing:

        - Strictly positive integers (>=1)

        - Elements sum to num_students

        - Random distribution between schools

    Example:
        > generate_school_capacities(100, 4)

        np.array([24, 31, 22, 23])  # Sum=100
    """
    capacities = np.random.choice(
        np.arange(1, num_students), size=num_schools - 1, replace=False
    )
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
    preferences = np.argsort(profiles, axis=1)[:, -1: -k - 1: -1]
    return preferences


def generate_possible_manipulations(
        num_schools: int, preferences: np.ndarray, k: int
) -> np.ndarray:
    """
    Generates possible preference manipulations for constrained DA (GS) by upgrading one school choice per position.

    Algorithm:
        1. For each position in original preferences:
        2. Find next available school not in original list
        3. Create new preference list with update and sort it
        4. Return unique new preference lists

    Parameters:
        num_schools (int): Total number of available schools
        preferences (np.ndarray): Original preference array (shape: (k,)),
                                 must be sorted in ascending order
        k (int): Length of preference lists (k <= num_schools)

    Returns:
        np.ndarray: Array of unique manipulated preference lists (shape: (m, k)),
                  sorted in ascending order per row

    Raises:
        ValueError: If preferences are not sorted or have invalid length

    Example:
        > prefs = np.array([0, 1])

        > generate_possible_manipulations(3, prefs, 2)

        np.array([[0, 2],
                  [1, 2]])
    """
    if len(preferences) != k or not np.all(np.diff(preferences) > 0):
        raise ValueError("Preferences must be a sorted array of length k.")

    preferences_set = set(preferences)
    manipulations = []

    for i in range(k):
        current_preference = preferences[i]

        # Find next available school
        for next_school in range(current_preference + 1, num_schools):
            if next_school not in preferences_set:
                new_prefs = preferences.copy()
                new_prefs[i] = next_school
                manipulations.append(np.sort(new_prefs))
                break

    return np.unique(manipulations, axis=0) if manipulations else np.array([])


def generate_possible_preferences_k_gs(num_schools: int, k: int) -> np.ndarray:
    """
    Generates all possible k-length school preference combinations in ascending order.

    Creates every possible combination of k schools from the total available schools,
    with each combination sorted in ascending numerical order.

    Parameters:
        num_schools (int): Total number of available schools
        k (int): Length of preference combinations to generate (k <= num_schools)

    Returns:
        numpy.ndarray: 2D array of shape (C(num_schools, k), k) containing:

        - Each row represents a unique school combination

        - Combinations are sorted in ascending order

        - School indices range from 0 to num_schools-1

    Example:
        > generate_possible_preferences_k_gs(4, 2)

        np.array([[0, 1],
               [0, 2],
               [0, 3],
               [1, 2],
               [1, 3],
               [2, 3]])
    """
    schools = np.arange(num_schools)
    combinations = list(itertools.combinations(schools, k))
    return np.array(combinations, dtype=np.int32)


def generate_possible_preferences_chinese(num_schools: int, k: int) -> np.ndarray:
    """
    Generates all possible school preference combinations using sequential k-length blocks.

    Creates permutations where:

    - Schools are selected in k-length blocks from remaining schools

    - Each block is sorted in ascending order

    - Process continues until all schools are assigned

    Parameters:
        num_schools (int): Total number of available schools
        k (int): Block size for school selection (k <= num_schools)

    Returns:
        numpy.ndarray: 2D array containing all valid preference combinations with:

        - Each row representing a complete preference ordering

        - Schools grouped in ascending-order blocks

        - All schools (0 to num_schools-1) included exactly once per row

    Example:
        > generate_possible_preferences_chinese(4, 2)

        np.array([
            [0, 1, 2, 3],  # Block 0-1 then 2-3 \n
            [0, 2, 1, 3],  # Block 0-2 then 1-3 \n
            [0, 3, 1, 2],  # Block 0-3 then 1-2 \n
            [1, 2, 0, 3],  # Block 1-2 then 0-3 \n
            [1, 3, 0, 2],  # Block 1-3 then 0-2 \n
            [2, 3, 0, 1]   # Block 2-3 then 0-1 \n
        ])
    """
    schools = np.arange(num_schools)
    all_combinations = []

    def generate_blocks(remaining_schools, current_pref):
        if len(remaining_schools) == 0:
            all_combinations.append(current_pref)
            return

        # Select k schools (or remaining if <k available)
        n = min(k, len(remaining_schools))
        for selected in itertools.combinations(remaining_schools, n):
            # Sort selected schools in ascending order
            sorted_selected = sorted(selected)
            new_remaining = [s for s in remaining_schools if s not in selected]
            generate_blocks(new_remaining, current_pref + list(sorted_selected))

    generate_blocks(list(schools), [])
    return np.array(all_combinations, dtype=np.int32)


def generate_symmetric_preferences(
        num_students: int,
        num_schools: int,
        k: int,
        profiles: np.ndarray,
        get_possible_preferences_func: Callable[[int, int], np.ndarray] = generate_possible_preferences_chinese
) -> list[np.ndarray]:
    """
    Generates symmetric variants of student preference lists for schools.
    For students with the same utility profiles, the same preferences are generated in each variant.
    The possible preferences depend on the mechanism for which they will be used.

    Parameters:
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        k (int): Block or list constraint size (mechanism dependent)
        profiles (np.ndarray): Matrix of utility profiles (shape: (num_students, num_schools))
        get_possible_preferences_func (Callable): Function generating valid school combinations,
            Default: generate_possible_preferences_k_gs

    Returns:
        list[np.ndarray]: List of preference matrices where:

        - Each matrix has shape (num_students, k)

        - Students with identical utilities get identical preferences

        - All combinations of group preferences are generated
    """
    # Group students by utility profiles
    groups = {}
    for i in range(num_students):
        key = tuple(profiles[i])
        groups.setdefault(key, []).append(i)

    # Generate possible preference combinations for each group
    possible_preferences = get_possible_preferences_func(num_schools, k)
    prefs_per_group = {u: possible_preferences for u in groups.keys()}

    # Generate combination matrix for unique groups
    group_keys = list(groups.keys())
    product_inputs = [prefs_per_group[key] for key in group_keys]

    # Compose final preference profiles
    profile_prefs = []
    for combo in itertools.product(*product_inputs):
        profile = [None] * num_students
        for group_idx, pref_array in enumerate(combo):
            student_indices = groups[group_keys[group_idx]]
            for student_idx in student_indices:
                profile[student_idx] = pref_array

        profile_prefs.append(np.array(profile))

    return profile_prefs