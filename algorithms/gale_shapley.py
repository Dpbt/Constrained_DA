import numpy as np


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