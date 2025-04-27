import numpy as np
import random
from utils import (generate_k_restricted_preferences, generate_possible_manipulations,
                   calculate_utilities_from_probs, calculate_utilities_from_probs_individual,
                   generate_school_capacities, generate_random_profiles,
                   generate_statistic)


random.seed(42)
np.random.seed(42)


def algorithm_sampler(algorithm: str, num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray,
                      k: int, num_repeat: int
                      ):
    # Повторяет указанный алгоритм на переданных списках num_repeat раз, возвращает вероятности.
    # statistic[:, num_schools] - сколько раз не попали никуда

    statistic = np.zeros((num_students, num_schools + 1))

    student_rank_default = list(range(num_students))
    def generate_random_permutations(elements):
        while True:
            yield random.sample(elements, len(elements))

    student_ranks_iterator = generate_random_permutations(student_rank_default)

    if algorithm == "boston":
        algorithm_func = k_boston_algorithm
    elif algorithm == 'gs':
        algorithm_func = k_gs_algorithm

    for i in range(num_repeat):
        current_school_preferences = next(student_ranks_iterator)

        assignments, unassigned_students = algorithm_func(num_students=num_students,
                                                          num_schools=num_schools,
                                                          preferences=preferences,
                                                          capacities=capacities,
                                                          k=k,
                                                          school_preferences=current_school_preferences)

        for school_key in assignments:
            for student in assignments[school_key]:
                statistic[student, school_key] += 1

        for student in unassigned_students:
            statistic[student, num_schools] += 1

    probabilities = statistic / num_repeat
    unassigned_statistic = statistic[:, num_schools] / num_repeat

    return probabilities, unassigned_statistic


def k_boston_algorithm(
    num_students: int,
    num_schools: int,
    preferences: np.ndarray,
    capacities: np.ndarray,
    k: int,
    school_preferences: tuple
) -> tuple[dict[int, list[int]], set[int]]:
    assignments = {school: [] for school in range(num_schools)}
    unassigned_students = set(range(num_students))

    for curr_round in range(1, k + 1):
        # print(curr_round, assignments, unassigned_students)
        for school in range(num_schools):
            current_applicants = [student for student in unassigned_students if preferences[student][curr_round - 1] == school]
            current_capacity = capacities[school] - len(assignments[school])

            if len(current_applicants) <= current_capacity:
                assignments[school].extend(current_applicants)
                for student in current_applicants:
                    unassigned_students.remove(student)

            else:
                sorted_current_applicants_list = sorted(
                    list(current_applicants), key=lambda x: school_preferences[x]
                )
                students_to_assign = sorted_current_applicants_list[: capacities[school]]

                # students_to_assign = np.random.choice(current_applicants, size=current_capacity, replace=False)
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
    school_preferences: tuple
) -> tuple[dict[int, list[int]], set[int]]:

    preferences_list = preferences.tolist()
    capacities_list = capacities.tolist()
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

                sorted_current_applicants_list = sorted(
                    list(all_current_applicants), key=lambda x: school_preferences[x]
                )
                students_to_assign = sorted_current_applicants_list[: capacities[school]]

                # random.shuffle(all_current_applicants_list)
                # students_to_assign = all_current_applicants_list[:capacities_list[school]]

                students_to_assign_set = set(students_to_assign)

                assignments[school] = students_to_assign_set
                unassigned_students -= students_to_assign_set
                unassigned_students.update(curr_assignments - students_to_assign_set)

        if not unassigned_students:
            break

    assignments = {school: list(assignments[school]) for school in range(num_schools)}
    return assignments, unassigned_students


def k_gs_algorithm_prob_individual(num_students: int, num_schools: int, preferences: np.ndarray,
                                   capacities: np.ndarray, k: int, student: int):
    # Оценка вероятности быть назначенным в каждую школу для ученика при алгоритме k_gs
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


def manipulation_algorithm(algorithm: str,
                           num_students: int,
                           num_schools: int,
                           profiles: np.ndarray,
                           capacities: np.ndarray,
                           k: int,
                           epsilon: float,
                           fair_indices: np.ndarray,
                           num_manipulations: int
                           ):
    if algorithm == 'gs':
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

            curr_probabilities = prob_func(num_students=num_students,
                                           num_schools=num_schools,
                                           preferences=preferences,
                                           capacities=capacities,
                                           k=k,
                                           student=student)
            curr_utilities = calculate_utilities_from_probs_individual(student=student,
                                                                       probabilities=curr_probabilities,
                                                                       profiles=profiles)

            best_manipulation = []
            best_manipulation_score = 0
            for new_preference in generate_possible_manipulations(num_schools, preferences[student], k):
                new_preferences = preferences.copy()
                new_preferences[student] = new_preference

                new_probabilities = prob_func(num_students=num_students,
                                              num_schools=num_schools,
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
    num_students = 10
    num_schools = 5
    k = 4
    profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
    print("profiles", profiles, sep='\n')
    capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
    print("capacities", capacities, sep='\n')
    preferences = generate_k_restricted_preferences(profiles, k)

    preferences[0] = np.array([1, 2, 3, 4])
    preferences[1] = np.array([1, 2, 3, 4])
    preferences[2] = np.array([0, 2, 3, 4])
    preferences[3] = np.array([0, 1, 3, 4])
    # print("preferences", preferences, sep='\

    probabilities, unassigned_statistic = algorithm_sampler(
        algorithm="gs",
        num_students=num_students,
        num_schools=num_schools,
        preferences=preferences,
        capacities=capacities,
        k=k,
        num_repeat=10000,
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
