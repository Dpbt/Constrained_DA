import numpy as np

from .gale_shapley import k_gs_algorithm


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