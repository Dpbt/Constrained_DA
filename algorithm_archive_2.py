import numpy as np
import random
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

    Parameters:
        algorithm (AlgorithmEnum): Matching mechanism to test (BOSTON_MECHANISM/K_GS_MECHANISM)
        num_students (int): Total number of students
        num_schools (int): Total number of available schools
        preferences (np.ndarray): Student preference matrix (shape: (num_students, k))
        capacities (np.ndarray): School capacities (shape: (num_schools,))
        k (int): Length of preference lists (k <= num_schools)
        num_repeat (int): Number of iterations of algorithm repetition to average the results

    Returns:
        tuple[np.ndarray, np.ndarray]: Two probability matrices:

        - Assignment probabilities (shape: (num_students, num_schools + 1))

        - Unassignment probabilities (shape: (num_students,))

    Raises:
        ValueError: If unknown algorithm type provided

    Implementation Notes:
        1. Randomization Process:
            - Implicit in algorithm_func (requires external permutation handling)
        2. Statistics Collection:
            - Tracks school assignments in first num_schools columns
            - Records unassignments in last column
        3. Probability Calculation:
            - Normalizes counts by num_repeat
            - Maintains separate unassignment statistics
    """
    statistic = np.zeros((num_students, num_schools + 1))

    # Algorithm Selection
    if algorithm == AlgorithmEnum.BOSTON_MECHANISM:
        algorithm_func = k_boston_algorithm
    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        algorithm_func = k_gs_algorithm
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}. "
                         f"Only AlgorithmEnum.K_GS_MECHANISM and "
                         f"AlgorithmEnum.BOSTON_MECHANISM are supported now.")

    # Simulation Loop
    for _ in range(num_repeat):
        assignments, unassigned = algorithm_func(
            num_students=num_students,
            num_schools=num_schools,
            preferences=preferences,
            capacities=capacities,
            k=k
        )

        # Update Statistics
        for school, students in assignments.items():
            for student in students:
                statistic[student, school] += 1
        for student in unassigned:
            statistic[student, num_schools] += 1

    # Normalization
    probabilities = statistic / num_repeat
    unassigned_probs = statistic[:, num_schools] / num_repeat

    return probabilities, unassigned_probs


def k_boston_algorithm(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int
                       ) -> tuple[dict[int, list[int]], set[int]]:
    assignments = {school: [] for school in range(num_schools)}
    unassigned_students = set(range(num_students))

    for curr_round in range(1, k + 1):
        # print(curr_round, assignments, unassigned_students)
        for school in range(num_schools):
            current_applicants = [student for student in unassigned_students if
                                  preferences[student][curr_round - 1] == school]
            current_capacity = capacities[school] - len(assignments[school])

            if len(current_applicants) <= current_capacity:
                assignments[school].extend(current_applicants)
                for student in current_applicants:
                    unassigned_students.remove(student)

            else:
                students_to_assign = np.random.choice(current_applicants, size=current_capacity, replace=False)
                assignments[school].extend(students_to_assign)
                for student in students_to_assign:
                    unassigned_students.remove(student)

        if (not unassigned_students or
                all(capacities[school] - len(assignments[school]) == 0 for school in range(num_schools))):
            break

    return assignments, unassigned_students


def k_gs_algorithm(
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
) -> tuple[dict[int, list[int]], set[int]]:
    preferences_list = preferences.tolist()
    capacities_list = capacities.tolist()
    # Реализация второго алгоритма распределения
    assignments = {school: set() for school in range(num_schools)}
    unassigned_students = set(range(num_students))
    curr_student_school = [0] * num_students
    num_applications = k * num_students
    num_iter = 0

    curr_student_school_sum = 0
    while curr_student_school_sum < num_applications:
        num_iter += 1

        for student in unassigned_students:
            curr_student_school[student] += 1
            curr_student_school_sum += 1

        for school in range(num_schools):
            current_applicants_set = set()
            for student in unassigned_students:
                if (
                        curr_student_school[student] <= k
                        and preferences_list[student][curr_student_school[student] - 1] == school
                ):
                    current_applicants_set.add(student)

            current_capacity = capacities_list[school] - len(assignments[school])

            if len(current_applicants_set) <= current_capacity:
                assignments[school].update(current_applicants_set)
                unassigned_students -= current_applicants_set

            else:
                curr_assignments = assignments[school]
                all_current_applicants = curr_assignments.union(current_applicants_set)
                all_current_applicants_list = list(all_current_applicants)

                random.shuffle(all_current_applicants_list)
                students_to_assign = all_current_applicants_list[:capacities_list[school]]
                students_to_assign_set = set(students_to_assign)

                assignments[school] = students_to_assign_set
                unassigned_students -= students_to_assign_set
                unassigned_students.update(curr_assignments - students_to_assign_set)

        if not unassigned_students:
            break

    assignments = {school: list(assignments[school]) for school in range(num_schools)}
    return assignments, unassigned_students


def k_gs_algorithm_prob_individual(num_schools: int, preferences: np.ndarray,
                                   capacities: np.ndarray, k: int, student: int):
    # Оценка вероятности быть назначенным в каждую школу для каждого ученика при алгоритме k_gs
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    # print(statistic)
    # probabilities = np.zeros(num_schools)
    probabilities = [0 for _ in range(num_schools)]

    curr_sum = 0

    for curr_preference in range(k):
        curr_school = preferences[student, curr_preference]
        # curr_prob = 1 - np.sum(probabilities[:curr_school])
        curr_prob = 1 - curr_sum

        num_competitors = 0

        for curr_step in range(k):
            avg_capacities = (np.sum(capacities[:curr_school]) / curr_school) * curr_step if curr_school > 0 else 0
            # print("avg:", np.sum(capacities[:curr_school]), curr_school, curr_step)
            curr_stats = np.sum(statistic[:curr_step, :curr_school])

            if curr_stats > 0:
                prob_assigned = avg_capacities / curr_stats
            else:
                prob_assigned = 0
            prob_unassigned = 1 - prob_assigned
            # print("otl:", curr_step, prob_assigned, prob_unassigned, avg_capacities, np.sum(statistic[:curr_step, :curr_school]), statistic[curr_step, curr_school])
            num_competitors += prob_unassigned * statistic[curr_step, curr_school]

        # print("fin:", student, curr_school, curr_preference, curr_prob, num_competitors, curr_prob * capacities[curr_school] / num_competitors)
        final_prob = curr_prob * capacities[curr_school] / num_competitors
        probabilities[curr_school] = final_prob
        curr_sum += final_prob

    # print(probabilities)
    # print(np.sum(probabilities, axis=1))
    return probabilities


def chinese_parallel_mechanism(
        num_students: int,
        num_schools: int,
        preferences: np.ndarray,
        capacities: np.ndarray,
        k: int,
        school_preferences: dict[int, int],
) -> tuple[dict[int, list[int]], set[int]]:
    """
    Реализация китайского параллельного механизма Ch^(k)

    Параметры:
    - num_students: количество студентов
    - num_schools: количество школ
    - preferences: матрица предпочтений студентов (student x school)
    - capacities: вектор вместимости школ
    - k: параметр механизма (размер блока)
    - school_preferences: матрица предпочтений школ (school x student)

    Возвращает:
    - Словарь {школа: список принятых студентов}
    """
    # Инициализация
    final_assignments = {school: [] for school in range(num_schools)}
    remaining_students = set(range(num_students))
    final_assignments_students = set()
    remaining_capacities = np.copy(capacities)
    round_num = 0

    while remaining_students and np.sum(remaining_capacities) > 0:
        # Подготовка предпочтений для текущего раунда
        curr_prefs = np.full((num_students, k), -1)

        for student in remaining_students:
            start = round_num * k
            end = (round_num + 1) * k
            # Берем текущий блок предпочтений или остаток, если блок меньше k
            block = preferences[student][start:end]
            # Если блок меньше k, дополняем -1 (no preference)
            if len(block) < k:
                block = np.append(block, [-1] * (k - len(block)))
            curr_prefs[student] = block

        # Запуск ограниченного алгоритма Гейла-Шепли для текущего раунда
        round_assignments, unmatched = k_gs_algorithm(
            num_students=num_students,
            num_schools=num_schools,
            preferences=curr_prefs,
            capacities=remaining_capacities,
            k=k,
            school_preferences=school_preferences,
        )

        # Обновляем финальные назначения и оставшиеся места
        for school in round_assignments:
            final_assignments[school].extend(round_assignments[school])
            remaining_capacities[school] -= len(round_assignments[school])

            for student in round_assignments[school]:
                # final_assignments_students.add(student)
                if student in remaining_students:
                    remaining_students.remove(student)

        # Обновляем множество оставшихся студентов
        for student in unmatched:
            if student not in final_assignments_students:
                remaining_students.add(student)

        # Переходим к следующему раунду
        round_num += 1

    unassigned_students = set()  # always empty for chinese mechanism

    return final_assignments, unassigned_students


def manipulation_algorithm(algorithm: AlgorithmEnum,
                           num_students: int,
                           num_schools: int,
                           profiles: np.ndarray,
                           capacities: np.ndarray,
                           k: int,
                           epsilon: float,
                           fair_indices: np.ndarray,
                           num_manipulations: int
                           ):
    if algorithm == AlgorithmEnum.K_GS_MECHANISM:
        prob_func = k_gs_algorithm_prob_individual
    # elif algorithm == 'boston':
    #     prob_func = k_boston_algorithm_prob
    else:
        raise ValueError('Algorithm must be only "gs" now')

    preferences = generate_k_restricted_preferences(profiles, k)

    manipulators = np.zeros(num_students)
    manipulators[fair_indices] = num_manipulations
    max_num_manipulations = num_manipulations * num_students

    last_manipulation = 1

    while np.sum(manipulators) < max_num_manipulations:
        if last_manipulation == 0:
            break
        else:
            last_manipulation = 0

        students_for_manipulation = [i for i in range(num_students) if manipulators[i] < num_manipulations]

        # Выбор порядка для манипулирования
        random.shuffle(students_for_manipulation)
        order_for_manipulation = students_for_manipulation

        # или
        # sorted_students = np.argsort(np.array(curr_utilities))
        # order_for_manipulation = [student for student in sorted_students if student in students_for_manipulation]
        # last_manipulation = 0

        for student in order_for_manipulation:

            curr_probabilities = prob_func(num_schools=num_schools,
                                           preferences=preferences,
                                           capacities=capacities,
                                           k=k,
                                           student=student)
            curr_utilities = calculate_utilities_from_probs_individual(student=student,
                                                                       probabilities=curr_probabilities,
                                                                       profiles=profiles)

            # print("For print", student)
            best_manipulation = []
            best_manipulation_score = 0
            for new_preference in generate_possible_manipulations(num_schools, preferences[student], k):
                new_preferences = preferences.copy()
                new_preferences[student] = new_preference

                new_probabilities = prob_func(num_schools=num_schools,
                                              preferences=new_preferences,
                                              capacities=capacities,
                                              k=k,
                                              student=student)
                new_utilities = calculate_utilities_from_probs_individual(student=student,
                                                                          probabilities=new_probabilities,
                                                                          profiles=profiles)

                manipulation_score = new_utilities - curr_utilities

                if manipulation_score > best_manipulation_score and manipulation_score > epsilon:
                    best_manipulation = new_preference
                    best_manipulation_score = manipulation_score

            if best_manipulation_score > 0:
                manipulators[student] += 1
                last_manipulation = 1
                preferences[student] = best_manipulation
                break

    manipulators[fair_indices] = 0
    return preferences, manipulators


if __name__ == '__main__':
    # start_time = time.time()
    # preferences = np.array([[0, 1], [0, 1], [0, 2]])
    # assignments, unassigned_students = k_gs_algorithm(num_students=3,
    #                                                   num_schools=3,
    #                                                   preferences=preferences,
    #                                                   capacities=np.array([1, 1, 1]),
    #                                                   k=2)
    #
    # print(assignments)
    # print(unassigned_students)
    # print(time.time() - start_time)

    # preferences = np.array([[0, 1, 2], [0, 1, 2], [1, 0, 2]])
    # probabilities = k_gs_algorithm_prob(3, 3, preferences, np.array([1, 1, 1]), 3)
    #
    # print(probabilities)

    num_students = 10
    num_schools = 5
    k = 4
    profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
    print("profiles", profiles, sep='\n')
    capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
    print("capacities", capacities, sep='\n')
    preferences = generate_k_restricted_preferences(profiles, k)
    # print("preferences", preferences, sep='\n')

    preferences[0] = np.array([1, 2, 3, 4])
    preferences[1] = np.array([1, 2, 3, 4])
    preferences[2] = np.array([0, 2, 3, 4])
    preferences[3] = np.array([0, 1, 3, 4])
    # print("preferences", preferences, sep='\n')

    probabilities, unassigned_statistic = algorithm_sampler(
        algorithm="gs",
        num_students=num_students,
        num_schools=num_schools,
        preferences=preferences,
        capacities=capacities,
        k=k,
        num_repeat=100000,
    )

    print(probabilities, unassigned_statistic)

    # assignments, unassigned_students = k_gs_algorithm(num_students, num_schools, preferences, capacities, k)
    #
    # print("assignments", assignments, sep='\n')
    # print("unassigned_students", unassigned_students, sep='\n')

    # manipulators_ratio = 1
    # num_fair = round(num_students * (1 - manipulators_ratio))
    # fair_indices = np.random.choice(num_students, num_fair, replace=False)
    #
    # preferences, manipulators = manipulation_algorithm(algorithm="gs",
    #                                                    num_students=num_students,
    #                                                    num_schools=num_schools,
    #                                                    profiles=profiles,
    #                                                    capacities=capacities,
    #                                                    k=k,
    #                                                    epsilon=0.01,
    #                                                    fair_indices=fair_indices,
    #                                                    num_manipulations=3)
    #
    # print("preferences", preferences, sep='\n')
    # print("manipulators", manipulators, sep='\n')
    #
    # probabilities, average_percentage_unassigned_students = algorithm_sampler(algorithm='gs',
    #                                                                           num_students=num_students,
    #                                                                           num_schools=num_schools,
    #                                                                           preferences=preferences,
    #                                                                           capacities=capacities,
    #                                                                           k=k,
    #                                                                           num_repeat=1000
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
