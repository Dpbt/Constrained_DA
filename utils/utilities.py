import numpy as np


def calculate_utilities(
        num_students: int, assignments: dict[int, list[int]], profiles: np.ndarray
) -> dict[int, float]:
    """
    Calculates individual utility for students based on their school assignments.

    Parameters:
        num_students (int): Total number of students
        assignments (dict[int, list[int]]): School assignments {school_id: [student_ids]}
        profiles (np.ndarray): Matrix of cardinal utility of students from schools
                              Shape: (num_students, num_schools), dtype: float

    Returns:
        dict[int, float]: Mapping {student_id: utility_score} where:
        - utility_score ∈ [0, 1]
        - Score represents preference strength for assigned school

    Example:
        > profiles = np.array([
             [0.9, 0.1],  # Student 0 \n
             [0.8, 0.2],  # Student 1 \n
             [0.7, 0.3]   # Student 2
          ])

        > assignments = {
             0: [1, 2],  # School 0 gets students 1 and 2 \n
             1: [0]      # School 1 gets student 0
          }

        > calculate_utilities(3, assignments, profiles)

        {0: 0.1, 1: 0.8, 2: 0.7}
    """
    student_utility = {i: 0.0 for i in range(num_students)}

    for school_id, students in assignments.items():
        for student in students:
            student_utility[student] = float(profiles[student, school_id])

    return student_utility


def calculate_utilities_from_probs(
        num_schools: int, probabilities: np.ndarray, profiles: np.ndarray
) -> np.ndarray:
    """
    Calculation of expected utilities for all students using school assignment probabilities.

    Parameters:
        num_schools (int): Total number of available schools
        probabilities (list[float]): Assignment probability matrix
            Shape: (num_students, num_schools+), each row sums to 1
        profiles (NDArray[float64]): Matrix of cardinal utility of students from schools
            Shape: (num_students, num_schools+), values normalized [0,1]

    Returns:
        np.ndarray[float64]: 1D array of expected utilities (shape: num_students)

    Example:
        > profiles = np.array([
             [0.9, 0.1, 0.0],  # Student 0 preferences \n
             [0.6, 0.3, 0.1]   # Student 1 preferences
        ], dtype=np.float64)

        > probabilities = np.array([
             [0.8, 0.2, 0.0],  # Student 0 assignment probs \n
             [0.1, 0.7, 0.2]   # Student 1 assignment probs
         ], dtype=np.float64)

        > calculate_utilities_from_probs(2, 3, probabilities, profiles)

        np.array([0.74 0.29])  # 0.9*0.8 + 0.1*0.2 = 0.74 | 0.6*0.1 + 0.3*0.7 + 0.1*0.2= 0.29
    """
    return np.sum(probabilities[:, :num_schools] * profiles[:, :num_schools], axis=1)


def calculate_utilities_from_probs_individual(
        student: int, probabilities: list[float], profiles: np.ndarray
) -> float:
    """
    Calculate expected utility for a single student using school assignment probabilities.

    Parameters:
        student (int): Index of the student (0-based)
        probabilities (list[float]): Assignment probability vector for the student
                                  Shape: (num_schools,) or (num_schools+), sum=1
        profiles (np.ndarray): Matrix of cardinal utility of students from schools
                             Shape: (num_students, num_schools+)

    Returns:
        float: Expected utility score

    Example:
        > profiles = np.array([
             [0.9, 0.1],  # Student 0 \n
             [0.6, 0.4]   # Student 1
        ])

        > probabilities = np.array([0.8, 0.2])  # Probabilities for student 0

        > calculate_utilities_from_probs_individual(0, 2, probs, profiles)

        0.74  # 0.9*0.8 + 0.1*0.2 = 0.74
    """
    return np.sum(probabilities * profiles[student])