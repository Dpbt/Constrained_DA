from collections import defaultdict

import numpy as np


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
