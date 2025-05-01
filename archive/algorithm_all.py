import numpy as np
import random
from time import time
from collections import defaultdict
from utils import (AlgorithmEnum, generate_k_restricted_preferences, generate_possible_manipulations,
                   calculate_utilities_from_probs, calculate_utilities_from_probs_individual,
                   generate_school_capacities, generate_random_profiles,
                   generate_statistic)

random.seed(42)
np.random.seed(42)


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


def k_boston_algorithm(
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
        school_preferences: dict[int, int]
) -> tuple[dict[int, list[int]], set[int]]:
    """
    Implements the Boston school choice mechanism with student preference lists of length k.

    Note: the current implementation only supports equal prioritization of schools with respect to students
    (school_preferences parameter)

    Parameters:
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        preferences (np.ndarray): Student preference matrix (shape: (num_students, k))
                                  Each row represents a student's school preferences per round
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        k (int): Length of preference lists (k <= num_schools)
        school_preferences (dict[int, int]): The order of priorities of all schools with respect to students.
                                             school_preferences[i] is the i-th student in the ranking

    Returns:
        tuple[dict[int, list[int]], set[int]]:
        - School assignments: Dictionary mapping school indices to lists of assigned students
        - Unassigned students: Set of student indices not assigned to any school

    Raises:
        ValueError:
            - If preferences matrix shape doesn't match (num_students, k)
            - If capacities array length doesn't match num_schools
            - If school_preferences length doesn't match num_students

    Implementation Notes:
        1. Round Processing:
            - Processes applications in k sequential rounds
            - Unassigned students apply to their current-round preferred school
        2. Assignment Logic:
            - Schools accept new applicants based on priority order
            - Fills remaining capacity each round
        3. Early Termination:
            - Stops processing rounds when no unassigned students remain

    Example:
        > prefs = np.array([[0, 1], [0, 1], [1, 2]])

        > caps = np.array([1, 1, 1])

        > school_prefs = (0, 1, 2)  # i.e., student 0 has first priority,
                                            student 1 has second priority,
                                            student 2 has third priority.

        > k_boston_algorithm(3, 3, prefs, caps, 2, school_prefs)

        ({0: [0], 1: [2], 2: []} {1})  # i.e., student 0 was assigned to school 0,
                                               student 2 was assigned to school 1,
                                               student 1 remained unassigned
    """
    preferences = preferences[:, :k]

    # Validate input dimensions
    if preferences.shape != (num_students, k):
        raise ValueError(f"Preferences matrix must be shape ({num_students}, {k})")
    if capacities.shape != (num_schools,):
        raise ValueError(f"Capacities array must be shape ({num_schools},)")
    if len(school_preferences) != num_students:
        raise ValueError(f"School preferences must contain {num_students} elements")

    # Initialize assignment storage and tracking sets
    assignments = {school: [] for school in range(num_schools)}
    unassigned_students = set(range(num_students))

    for round_num in range(k):
        if not unassigned_students:
            break  # Early exit if all students placed

        # Collect applications for current round
        round_applications = defaultdict(list)
        for student in unassigned_students:
            preferred_school = preferences[student, round_num]
            round_applications[preferred_school].append(student)

        # Process applications for each school
        for school in range(num_schools):
            # Calculate remaining capacity
            remaining_capacity = capacities[school] - len(assignments[school])
            if remaining_capacity <= 0:
                continue

            # Sort applicants by school priority and select top candidates
            applicants = round_applications.get(school, [])
            if applicants:
                # Sort using cached priorities (lower rank = higher priority)
                sorted_applicants = sorted(applicants, key=lambda x: school_preferences[x])
                selected = sorted_applicants[:remaining_capacity]

                # Update assignments and unassigned set
                assignments[school].extend(selected)
                unassigned_students.difference_update(selected)

    return assignments, unassigned_students


def k_gs_algorithm(
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
        school_preferences: dict[int, int]
) -> tuple[dict[int, list[int]], set[int]]:
    """
        Implements the constrained Gale-Shapley school choice mechanism with student preference lists of length k.

        Note: the current implementation only supports equal prioritization of schools with respect to students
        (school_preferences parameter)

        Parameters:
            num_students (int): Total number of students
            num_schools (int): Total number of available schools
            preferences (np.ndarray): Student preference matrix (shape: (num_students, k))
                                      Each row represents a student's school preferences per round
            capacities (np.ndarray): School capacities (shape: (num_schools,))
            k (int): Length of preference lists (k <= num_schools)
            school_preferences (dict[int, int]): The order of priorities of all schools with respect to students.
                                             school_preferences[i] is the i-th student in the ranking

        Returns:
            tuple[dict[int, list[int]], set[int]]:
            - School assignments: Dictionary mapping school indices to lists of assigned students
            - Unassigned students: Set of student indices not assigned to any school

        Raises:
            ValueError:
                - If preferences matrix shape doesn't match (num_students, k)
                - If capacities array length doesn't match num_schools
                - If school_preferences length doesn't match num_students

        Implementation Notes:
            1. Round Processing:
                - A round is held if there are unassigned students who have not exhausted their lists.
                - Unassigned students apply to their current-round preferred school
            2. Assignment Logic:
                - Schools admit students according to the order of priority between current assigned students and new applicants.
            3. Early Termination:
                - Stops processing rounds when no unassigned students remain

        Example:
            > prefs = np.array([[0, 1], [0, 1], [1, 2]])

            > caps = np.array([1, 1, 1])

            > school_prefs = (0, 1, 2)  # i.e., student 0 has first priority,
                                                student 1 has second priority,
                                                student 2 has third priority.

            > k_gs_algorithm(3, 3, prefs, caps, 2, school_prefs)

            ({0: [0], 1: [1], 2: [2]} set())  # i.e., student 0 was assigned to school 0,
                                                      student 1 was assigned to school 1,
                                                      student 2 was assigned to school 2
    """
    preferences = preferences[:, :k]

    # Validate input dimensions
    if preferences.shape != (num_students, k):
        raise ValueError(f"Preferences matrix must be shape ({num_students}, {k})")
    if capacities.shape != (num_schools,):
        raise ValueError(f"Capacities array must be shape ({num_schools},)")
    if len(school_preferences) != num_students:
        raise ValueError(f"School preferences must contain {num_students} elements")

    # Convert to lists for faster iteration
    preferences_list = preferences.tolist()
    capacities_list = capacities.tolist()

    # Initialize assignment storage and tracking sets
    assignments = {school: set() for school in range(num_schools)}
    unassigned_students = set(range(num_students))
    curr_round = [0] * num_students  # Track current application round per student

    # Main processing loop
    while any(curr_round[unassigned_student] < k for unassigned_student in unassigned_students) and unassigned_students:

        # Update application rounds for unassigned students
        for student in list(unassigned_students):
            if curr_round[student] < k:
                curr_round[student] += 1

        # Process applications per school
        for school in range(num_schools):
            # Collect current applicants
            current_applicants = {
                student for student in unassigned_students
                if (curr_round[student] <= k and
                    preferences_list[student][curr_round[student] - 1] == school)
            }

            if not current_applicants:
                continue

            # Combined pool of current and previous applicants
            all_candidates = assignments[school].union(current_applicants)

            # Sort by cached priority
            sorted_candidates = sorted(
                all_candidates,
                key=lambda x: school_preferences[x]
            )

            # Select top candidates
            selected = set(sorted_candidates[:capacities_list[school]])
            rejected = all_candidates - selected

            # Update assignments
            assignments[school] = selected
            unassigned_students -= selected
            unassigned_students.update(rejected)

            # Early exit if all students placed
            if not unassigned_students:
                break

    # Convert sets to lists for output
    return {s: list(students) for s, students in assignments.items()}, unassigned_students


def chinese_parallel_mechanism(
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
        school_preferences: dict[int, int]
) -> tuple[dict[int, list[int]], set[int]]:
    """
    Implements the Chinese parallel mechanism Ch^(k) with multi-round deferred acceptance, 
    each round with student preference lists of length k.

    Note: the current implementation only supports equal prioritization of schools with respect to students
    (school_preferences parameter)

    Parameters:
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        preferences (np.ndarray): Student preference matrix (shape: (num_students, num_schools))
                                  Each row represents a student's school preferences per round
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        k (int): Length of preference lists (k <= num_schools)
        school_preferences (dict[int, int]): The order of priorities of all schools with respect to students.
                                         school_preferences[i] is the i-th student in the ranking

    Returns:
        tuple[dict[int, list[int]], set[int]]:
        - School assignments: Dictionary mapping school indices to lists of assigned students
        - Unassigned students: Set of student indices not assigned to any school

    Raises:
        ValueError:
            - If preferences.shape != (num_students, m) where m >= k
            - If capacities.shape != (num_schools,)
            - If school_preferences length != num_students

    Implementation Notes:
        1. Round Processing:
            - Divides preferences into k-sized blocks for each round
            - A round is held if there are unassigned students
            - Students who find a school after the next round are assigned to it permanently
        2. Assignment Logic:
            - Schools admit students according to the order of priority between current assigned students (from this round) and new applicants.
        3. Early Termination:
            - Stops processing rounds when no unassigned students remain

    Example:
        > prefs = np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0]])

        > caps = np.array([1, 1, 1])

        > school_prefs = (2, 1, 0)  # i.e., student 2 has first priority,
                                            student 1 has second priority,
                                            student 0 has third priority.

        > chinese_parallel_mechanism(3, 3, prefs, caps, 2, school_prefs)

        ({0: [1], 1: [2], 2: [0]} set())  # i.e., student 1 was assigned to school 0,
                                                  student 2 was assigned to school 1,
                                                  student 0 was assigned to school 2
    """
    # Validate input dimensions
    if preferences.shape != (num_students, num_schools):
        raise ValueError(f"Preferences matrix must be shape ({num_students}, {num_schools})")
    if capacities.shape != (num_schools,):
        raise ValueError(f"Capacities must be shape ({num_schools},)")
    if len(school_preferences) != num_students:
        raise ValueError(f"School preferences must contain {num_students} elements")

    # Initialization
    final_assignments = {school: [] for school in range(num_schools)}
    unassigned_student = set(range(num_students))
    remaining_capacities = np.copy(capacities)
    round_num = 0

    # Main processing loop
    while unassigned_student and np.sum(remaining_capacities) > 0:
        curr_prefs = np.full(shape=(num_students, k), fill_value=-1)

        start = round_num * k
        block_size = min(k, preferences.shape[1] - start)
        if block_size > 0:
            for student in unassigned_student:
                student_block = preferences[student, start:start + block_size]
                curr_prefs[student, :block_size] = student_block

        # Run k-GS algorithm for current block
        round_assignments, _ = k_gs_algorithm(
            num_students=num_students,
            num_schools=num_schools,
            preferences=curr_prefs,
            capacities=remaining_capacities,
            k=k,
            school_preferences=school_preferences
        )

        # Update assignments and capacities
        for school, students in round_assignments.items():
            final_assignments[school].extend(students)
            remaining_capacities[school] -= len(students)
            unassigned_student.difference_update(students)

        round_num += 1

    return final_assignments, set()


def k_gs_algorithm_prob_individual(num_schools: int,
                                   preferences: np.ndarray,
                                   capacities: np.ndarray,
                                   k: int,
                                   student: int) -> list[float]:
    """
        Estimates the probabilities of assigning a student in a constrained Deferred Acceptance mechanism
        based on the statistics of the distribution of schools by place on student preference lists.

        Parameters:
            num_schools (int): Total number of available schools
            preferences (np.ndarray): Student preference matrix (shape: (num_students, k))
                                      Each row represents a student's school preferences per round
            capacities (np.ndarray): School capacities (shape: (num_schools,))
            k (int): Length of preference lists (k <= num_schools)
            student (int): Target student index for probability estimation

        Returns:
            list[float]: Probability distribution over schools (indexed 0 to num_schools-1)
        """
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    probabilities = [0 for _ in range(num_schools)]

    curr_sum = 0

    for curr_preference in range(k):
        curr_school = preferences[student, curr_preference]
        curr_prob = 1 - curr_sum

        num_competitors = 0

        for curr_step in range(k):
            avg_capacities = (np.sum(capacities[:curr_school]) / curr_school) * curr_step if curr_school > 0 else 0
            curr_stats = np.sum(statistic[:curr_step, :curr_school])

            if curr_stats > 0:
                prob_assigned = avg_capacities / curr_stats
            else:
                prob_assigned = 0
            prob_unassigned = 1 - prob_assigned
            num_competitors += prob_unassigned * statistic[curr_step, curr_school]

        final_prob = curr_prob * capacities[curr_school] / num_competitors
        probabilities[curr_school] = final_prob
        curr_sum += final_prob

    return probabilities


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


if __name__ == '__main__':
    num_students = 10
    num_schools = 5
    k = 4
    profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
    print("profiles", profiles, sep='\n')
    capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
    print("capacities", capacities, sep='\n')
    # preferences = generate_k_restricted_preferences(profiles, k)
    preferences = generate_k_restricted_preferences(profiles, num_schools)

    preferences[0] = np.array([1, 2, 3, 4, 0])
    preferences[1] = np.array([1, 2, 3, 4, 0])
    preferences[2] = np.array([0, 2, 3, 4, 1])
    preferences[3] = np.array([0, 1, 3, 4, 2])
    # print("preferences", preferences, sep='\)

    # school_preferences = tuple(range(num_students))
    # school_preferences = [1, 0, 4, 9, 6, 5, 8, 2, 3, 7]
    # print(school_preferences)
    #
    # assignments, unassigned_students = k_gs_algorithm(
    #     num_students=num_students,
    #     num_schools=num_schools,
    #     preferences=preferences,
    #     capacities=capacities,
    #     k=k,
    #     school_preferences=school_preferences
    # )
    #
    # print("assignments", assignments, sep='\n')
    # print("unassigned_students", unassigned_students, sep='\n')
    #
    # assignments, unassigned_students = k_gs_algorithm_new(
    #     num_students=num_students,
    #     num_schools=num_schools,
    #     preferences=preferences,
    #     capacities=capacities,
    #     k=k,
    #     school_preferences=school_preferences
    # )
    #
    # print("assignments", assignments, sep='\n')
    # print("unassigned_students", unassigned_students, sep='\n')

    # st = time()
    # probabilities, unassigned_statistic = algorithm_sampler(
    #     algorithm=AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
    #     num_students=num_students,
    #     num_schools=num_schools,
    #     preferences=preferences,
    #     capacities=capacities,
    #     k=k,
    #     num_repeat=100000,
    # )
    #
    # print("Time", time() - st)
    #
    # print(probabilities, unassigned_statistic)

    # prefs = np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0]])
    # caps = np.array([1, 1, 1])
    # school_pref = (2, 1, 0)  # Student 2 has highest priority
    # school_pref = {student: rank for rank, student in enumerate(school_pref)}
    #
    # assignments, unassigned_students = k_boston_algorithm(3, 3, prefs, caps, 2, school_pref)
    # print(assignments, unassigned_students)
    #
    # assignments, unassigned_students = k_gs_algorithm(3, 3, prefs, caps, 2, school_pref)
    # print(assignments, unassigned_students)
    #
    # assignments, unassigned_students = chinese_parallel_mechanism(3, 3, prefs, caps, 2, school_pref)
    # print(assignments, unassigned_students)

    num_students = 60
    num_schools = 9
    k = 7
    num_manipulations = 3
    epsilon = 0.005

    time_1 = 0
    time_2 = 0

    for i in range(100):
        print(i)
        # print("----------------------------------------------------------------------------------------------")
        profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
        capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
        manipulators_ratio = 0.75
        num_fair = round(num_students * (1 - manipulators_ratio))
        fair_indices = np.random.choice(num_students, num_fair, replace=False)

        random.seed(42)
        np.random.seed(42)

        st = time()
        preferences_1, manipulators_1 = manipulation_algorithm_new(algorithm=AlgorithmEnum.K_GS_MECHANISM,
                                                                   num_students=num_students,
                                                                   num_schools=num_schools,
                                                                   profiles=profiles,
                                                                   capacities=capacities,
                                                                   k=k,
                                                                   epsilon=epsilon,
                                                                   fair_indices=fair_indices,
                                                                   num_manipulations=num_manipulations)
        time_1 += (time() - st)

        # print("preferences", preferences_1, sep='\n')
        # print("preferences", type(preferences_1), sep='\n')
        # print("manipulators", manipulators_1, sep='\n')
        # print("manipulators", type(manipulators_1), sep='\n')

        random.seed(42)
        np.random.seed(42)

        st = time()

        preferences_2, manipulators_2 = manipulation_algorithm_new(algorithm=AlgorithmEnum.K_GS_MECHANISM,
                                                                   num_students=num_students,
                                                                   num_schools=num_schools,
                                                                   profiles=profiles,
                                                                   capacities=capacities,
                                                                   k=k,
                                                                   epsilon=epsilon,
                                                                   fair_indices=fair_indices,
                                                                   num_manipulations=num_manipulations)

        time_2 += (time() - st)
        # print("preferences", preferences_2, sep='\n')
        # print("manipulators", manipulators_2, sep='\n')

        if not np.array_equal(preferences_1, preferences_2) or not np.array_equal(manipulators_1, manipulators_2):
            print("@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("preferences_1", preferences_1, sep='\n')
            print("preferences_2", preferences_2, sep='\n')
            print("manipulators_1", manipulators_1, sep='\n')
            print("manipulators_2", manipulators_2, sep='\n')
            break

    print(time_1, time_2, sep='\n')

    # probabilities, average_percentage_unassigned_students = algorithm_sampler(algorithm='gs',
    #                                                                           num_students=num_students,
    #                                                                           num_schools=num_schools,
    #                                                                           preferences=preferences,
    #                                                                           capacities=capacities,
    #                                                                           k=k,
    #                                                                           num_repeat=10000
    #                                                                           )
    #
    # print("true probabilities", probabilities, sep='\n')
    #
    # utilities = calculate_utilities_from_probs(num_students=num_students,
    #                                           num_schools=num_schools,
    #                                           probabilities=probabilities,
    #                                           profiles=profiles)
    #
    # print("true utilities", utilities, sep='\n')
