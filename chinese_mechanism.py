import numpy as np
import itertools
from algorithm_modified import k_gs_algorithm, chinese_parallel_mechanism
from utils import AlgorithmEnum
from typing import Callable


# utils.py
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


# utils.py
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


# utils.py
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
        profiles (np.ndarray): Utility profiles matrix (shape: (num_students, num_schools))
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


# utils.py ?
def find_nash_equilibrium(
        results: list[tuple[np.ndarray, np.ndarray]],
        profiles: np.ndarray = None,
        symmetric: bool = False
) -> list[tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]]:
    """
    Identifies Nash equilibria in strategic games with optional symmetry constraints.

    Analyzes game outcomes to find strategy profiles where no player/group can
    unilaterally improve their payoff. Supports both standard and symmetric equilibria
    through utility-based player grouping.

    Parameters:
        results (list): Game outcomes as [(strategy_profile, utilities)] where:
            - strategy_profile: np.ndarray of player choices (shape: (players, strategies))
            - utilities: np.ndarray of corresponding payoffs (shape: (players,))
        profiles (np.ndarray): Utility profiles matrix (shape: (players, num_utilities))
            Required when symmetric=True
        symmetric (bool): Restrict to equilibria where identical-utility players
            use identical strategies

    Returns:
        list: Nash equilibria as [(strategy_profile, utilities)] where:
            - strategy_profile: Tuple of player strategy tuples
            - utilities: Corresponding payoff tuple

    Raises:
        ValueError: When symmetric=True and profiles=None
    """
    utility_dict = {}
    # Create utility dictionary with immutable types
    for pref, utils in results:
        strategy_profile = tuple(
            tuple(int(x) for x in player_strategy) for player_strategy in pref
        )
        utilities = tuple(float(u) for u in utils)
        utility_dict[strategy_profile] = utilities

    # Determine game parameters
    num_players = len(next(iter(utility_dict.keys())))
    players_strategies = []

    # Build strategy spaces
    for player_idx in range(num_players):
        strategies = {tuple(int(x) for x in pref[player_idx]) for pref, _ in results}
        players_strategies.append(list(strategies))

    # Player grouping logic
    utils_groups = {}
    if symmetric:
        if profiles is None:
            raise ValueError("profiles parameter is required when symmetric=True")

        for player_idx, prefs in enumerate(profiles):
            key = tuple(int(x) for x in prefs)
            utils_groups.setdefault(key, []).append(player_idx)

    def is_nash_equilibrium(strategy_profile):
        """Core equilibrium verification logic"""
        current_utilities = utility_dict[strategy_profile]

        if symmetric:
            # Group-based deviation check
            for group_indices in utils_groups.values():
                group_strategy = strategy_profile[group_indices[0]]

                if any(strategy_profile[i] != group_strategy for i in group_indices):
                    continue

                for candidate_strategy in players_strategies[group_indices[0]]:
                    if candidate_strategy == group_strategy:
                        continue

                    # Simulate group deviation
                    new_profile = list(strategy_profile)
                    for i in group_indices:
                        new_profile[i] = candidate_strategy
                    new_profile = tuple(new_profile)

                    if new_profile not in utility_dict:
                        continue

                    if all(utility_dict[new_profile][i] > current_utilities[i] for i in group_indices):
                        return False
        else:
            # Individual deviation check
            for player_idx in range(num_players):
                for candidate_strategy in players_strategies[player_idx]:
                    if candidate_strategy == strategy_profile[player_idx]:
                        continue

                    new_profile = list(strategy_profile)
                    new_profile[player_idx] = candidate_strategy
                    new_profile = tuple(new_profile)

                    if new_profile not in utility_dict:
                        continue

                    if utility_dict[new_profile][player_idx] > current_utilities[player_idx]:
                        return False
        return True

    # Profile generation and validation
    nash_equilibria = []
    for strategy_profile in itertools.product(*players_strategies):
        if strategy_profile not in utility_dict:
            continue

        # Symmetry validation
        valid_profile = True
        if symmetric:
            for group_indices in utils_groups.values():
                if len({strategy_profile[i] for i in group_indices}) != 1:
                    valid_profile = False
                    break

        if valid_profile and is_nash_equilibrium(strategy_profile):
            nash_equilibria.append((strategy_profile, utility_dict[strategy_profile]))

    # Console reporting
    if not nash_equilibria:
        print("No Nash equilibria found.")
    else:
        print(f"Found {len(nash_equilibria)} Nash equilibria ({'symmetric' if symmetric else 'all types'}):")
        for profile, utils in nash_equilibria:
            players_str = ", ".join(f"Player {i + 1}: {strat}" for i, strat in enumerate(profile))
            utils_str = ", ".join(f"{u:.2f}" for u in utils)
            print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


def all_preferences_test(
        num_schools: int,
        num_students: int,
        k: int,
        capacities: np.ndarray,
        profiles: np.ndarray,
        algorithm: AlgorithmEnum = AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Exhaustively tests all symmetric preference combinations against school permutations.

    Conducts full factorial analysis of student-school matching by evaluating:
    - All possible symmetric student preference profiles
    - Every school preference permutation
    - Specified matching algorithm behavior

    Parameters:
        num_schools (int): Total number of available schools
        num_students (int): Total number of students
        k (int): Block or list constraint size (mechanism dependent)
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        profiles (np.ndarray): Student utility profiles (shape: (num_students, num_schools))
        algorithm (AlgorithmEnum): Matching mechanism to test. Available options: AlgorithmEnum.CHINESE_PARALLEL_MECHANISM, AlgorithmEnum.K_GS_MECHANISM

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Test results containing:
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
    get_possible_preferences_func, algorithm_func = None, None
    if algorithm == AlgorithmEnum.CHINESE_PARALLEL_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_chinese
        algorithm_func = chinese_parallel_mechanism
        elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_k_gs
        algorithm_func = k_gs_algorithm
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

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

    results = []
    for i, preferences in enumerate(all_symmetric_preferences):
        preferences_utils = np.zeros(num_students)

        # Test against all school preference orders
        for school_preference in school_preferences:
            school_assignments, unassigned = algorithm_func(
                num_students=num_students,
                num_schools=num_schools,
                preferences=preferences,
                capacities=capacities,
                k=k,
                school_preferences=school_preference,
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


if __name__ == "__main__":
    # num_schools = 4
    # num_students = 4
    # k = 2
    # capacities = np.array([1, 1, 1, 1, 1])

    profiles = np.array([[54, 23, 15, 8], [54, 23, 15, 8], [38, 32, 30, 0], [38, 32, 30, 0]])

    # profiles = np.array([
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 4, 1],
    # ])

    # profiles = np.array([
    #     [90, 6, 4, 0],
    #     [90, 8, 2, 0],
    #     [90, 9, 1, 0],
    #     [90, 10, 0, 0],
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
        # algorithm=AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
        algorithm=AlgorithmEnum.K_GS_MECHANISM,
    )

    find_nash_equilibrium(results=results, profiles=profiles, symmetric=True)
