import time
import numpy as np
from utils import (generate_subsets, generate_k_restricted_preferences, generate_possible_manipulations,
                   calculate_utilities_from_prob, calculate_utilities_from_prob_individual,
                   generate_school_capacities, generate_random_profiles, generate_statistic)
import random


# Не нужен
def k_boston_algorithm_expectation(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int
                       ) -> tuple[dict[int, list[int]], set[int]]:
    assignments = {school: 0 for school in range(num_schools)}
    probabilities = np.zeros((num_students, num_schools))

    for curr_round in range(1, k + 1):
        current_probabilities = np.sum(probabilities[:, :curr_round - 1], axis=1)
        # print(curr_round, assignments, unassigned_students)
        print("curr_round", curr_round, current_probabilities)

        for school in range(num_schools):
            current_applicants = [student for student in range(num_students) if
                                  preferences[student][curr_round - 1] == school]
            current_capacity = capacities[school] - assignments[school]

            print("school", school, current_applicants, current_capacity)

            if len(current_applicants) <= current_capacity:
                assignments[school] += len(current_applicants)
                for student in current_applicants:
                    probabilities[student][curr_round - 1] = current_probabilities[student]

            else:
                assignments[school] = capacities[school]
                for student in current_applicants:
                    student_probability = 0
                    # other_applicants = current_applicants.remove(student)
                    other_applicants = [st for st in current_applicants if st != student]
                    print("student", student, other_applicants)
                    for num_competitors in range(len(current_applicants)):
                        print("num_competitors", num_competitors)
                        local_student_probability = 0
                        for curr_competitors in generate_subsets(other_applicants, num_competitors):
                            curr_competitors.append(student)
                            not_curr_competitors = [st for st in other_applicants if st not in curr_competitors]
                            print("curr_competitors", curr_competitors)
                            print("not_curr_competitors", not_curr_competitors)
                            local_student_probability += (np.prod(1 - current_probabilities[curr_competitors]) *
                                                          np.prod(current_probabilities[not_curr_competitors]))
                        if num_competitors + 1 <= current_capacity:
                            student_probability += local_student_probability
                        else:
                            student_probability += local_student_probability * current_capacity / (num_competitors + 1)

                    probabilities[student][curr_round - 1] = student_probability
                    print()
            print()
        print()
    return probabilities


def algorithm_sampler(algorithm: str, num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray,
                      k: int, num_repeat: int
                      ):
    # Повторяет указанный алгоритм на переданных списках num_repeat раз, возвращает вероятности.
    # statistic[:, num_schools] - сколько раз не попали никуда

    statistic = np.zeros((num_students, num_schools + 1))

    if algorithm == 'boston':
        algorithm_func = k_boston_algorithm
    elif algorithm == 'gs':
        algorithm_func = k_gs_algorithm

    for i in range(num_repeat):
        assignments, unassigned_students = algorithm_func(num_students=num_students,
                                                          num_schools=num_schools,
                                                          preferences=preferences,
                                                          capacities=capacities,
                                                          k=k)

        for key in assignments:
            for student in assignments[key]:
                statistic[student, key] += 1

        for student in unassigned_students:
            statistic[student, num_schools] += 1

    probabilities = statistic / num_repeat
    unassigned_statistic = statistic[:, num_schools] / num_repeat
    # average_percentage_unassigned_students = (np.sum(statistic[:, num_schools]) / num_repeat) / num_students * 100

    # Может добавить усреднение по одинаковым предпочтениям ?
    # return probabilities, average_percentage_unassigned_students
    return probabilities, unassigned_statistic


def k_boston_algorithm(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int
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
                students_to_assign = np.random.choice(current_applicants, size=current_capacity, replace=False)
                assignments[school].extend(students_to_assign)
                for student in students_to_assign:
                    unassigned_students.remove(student)

        if (not unassigned_students or
                all(capacities[school] - len(assignments[school]) == 0 for school in range(num_schools))):
            break

    return assignments, unassigned_students


def k_gs_algorithm(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int
                   ) -> tuple[dict[int, list[int]], set[int]]:
    # Реализация второго алгоритма распределения
    assignments = {school: [] for school in range(num_schools)}
    unassigned_students = set(range(num_students))
    curr_student_school = np.array([0 for _ in range(num_students)])
    num_applications = k * num_students
    num_iter = 0

    while np.sum(curr_student_school) < num_applications:
        num_iter += 1
        # print("new_iter", assignments, unassigned_students, curr_student_school)

        for student in unassigned_students:
            curr_student_school[student] += 1

        for school in range(num_schools):
            current_applicants = []
            for student in unassigned_students:
                # print(preferences, curr_student_school, student, curr_student_school[student] - 1, k)
                if curr_student_school[student] <= k and preferences[student][curr_student_school[student] - 1] == school:
                    current_applicants.append(student)

            current_capacity = capacities[school] - len(assignments[school])
            # print("school", school, current_applicants, current_capacity)

            if len(current_applicants) <= current_capacity:
                assignments[school].extend(current_applicants)
                for student in current_applicants:
                    unassigned_students.remove(student)
                    # curr_student_school[student] += 1

            else:
                curr_assignments = assignments[school]
                all_current_applicants = curr_assignments + current_applicants
                students_to_assign = np.random.choice(all_current_applicants, size=capacities[school], replace=False)
                assignments[school] = list(students_to_assign)
                # print("too much", assignments[school], all_current_applicants, students_to_assign, set(curr_assignments), set(students_to_assign))
                for student in students_to_assign:
                    unassigned_students.discard(student)
                    # curr_student_school[student] += 1
                for student in set(curr_assignments) - set(students_to_assign):
                    unassigned_students.add(student)

                # print("after_check", assignments, unassigned_students, curr_student_school)

        if not unassigned_students:
            break

    # print(num_iter)

    return assignments, unassigned_students


def k_gs_algorithm_prob(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int):
    # Оценка вероятности быть назначенным в каждую школу для каждого ученика при алгоритме k_gs
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    probabilities = np.zeros((num_students, num_schools))

    for student in range(num_students):
        for curr_preference in range(k):
            curr_school = preferences[student, curr_preference]
            if curr_preference > 0:
                # print(student, pref, curr_school)
                curr_competitors = min(capacities[curr_school], np.sum(statistic[:curr_preference, curr_school]))
            else:
                curr_competitors = 0
            probabilities[student][curr_school] = min(capacities[curr_school] / (statistic[curr_preference, curr_school] + curr_competitors), 1)

    return probabilities


def k_gs_algorithm_prob_2(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int):
    # Оценка вероятности быть назначенным в каждую школу для каждого ученика при алгоритме k_gs
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    # print(statistic)
    probabilities = np.zeros((num_students, num_schools))

    for student in range(num_students):
        for curr_preference in range(k):
            curr_school = preferences[student, curr_preference]
            curr_prob = 1 - np.sum(probabilities[student, :curr_school])

            num_competitors = 0

            for curr_step in range(k):
                avg_capacities = (np.sum(capacities[:curr_school]) / curr_school) * curr_step if curr_school > 0 else 0
                # print("avg:", np.sum(capacities[:curr_school]), curr_school, curr_step)
                prev_competitors = np.sum(statistic[:curr_step, :curr_school])
                if prev_competitors > 0:
                    prob_assigned = avg_capacities / np.sum(statistic[:curr_step, :curr_school])
                else:
                    prob_assigned = 0
                prob_unassigned = 1 - prob_assigned
                # print("otl:", curr_step, prob_assigned, prob_unassigned, avg_capacities, np.sum(statistic[:curr_step, :curr_school]), statistic[curr_step, curr_school])
                num_competitors += prob_unassigned * statistic[curr_step, curr_school]

            # print("fin:", student, curr_school, curr_preference, curr_prob, num_competitors, curr_prob * capacities[curr_school] / num_competitors)

            probabilities[student][curr_school] = curr_prob * capacities[curr_school] / num_competitors

    # print(probabilities)
    # print(np.sum(probabilities, axis=1))
    return probabilities


def k_gs_algorithm_prob_individual(num_students: int, num_schools: int, preferences: np.ndarray,
                                   capacities: np.ndarray, k: int, student: int):
    # Оценка вероятности быть назначенным в каждую школу для каждого ученика при алгоритме k_gs
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    # print(statistic)
    probabilities = np.zeros(num_schools)

    for curr_preference in range(k):
        curr_school = preferences[student, curr_preference]
        curr_prob = 1 - np.sum(probabilities[:curr_school])

        num_competitors = 0

        for curr_step in range(k):
            avg_capacities = (np.sum(capacities[:curr_school]) / curr_school) * curr_step if curr_school > 0 else 0
            # print("avg:", np.sum(capacities[:curr_school]), curr_school, curr_step)
            prev_competitors = np.sum(statistic[:curr_step, :curr_school])
            if prev_competitors > 0:
                prob_assigned = avg_capacities / np.sum(statistic[:curr_step, :curr_school])
            else:
                prob_assigned = 0
            prob_unassigned = 1 - prob_assigned
            # print("otl:", curr_step, prob_assigned, prob_unassigned, avg_capacities, np.sum(statistic[:curr_step, :curr_school]), statistic[curr_step, curr_school])
            num_competitors += prob_unassigned * statistic[curr_step, curr_school]

        # print("fin:", student, curr_school, curr_preference, curr_prob, num_competitors, curr_prob * capacities[curr_school] / num_competitors)

        probabilities[curr_school] = curr_prob * capacities[curr_school] / num_competitors

    # print(probabilities)
    # print(np.sum(probabilities, axis=1))
    return probabilities


# TODO if needed
def k_boston_algorithm_prob(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int):
    pass


def manipulation_algorithm_2(algorithm: str,
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
    elif algorithm == 'boston':
        prob_func = k_boston_algorithm_prob
    else:
        raise ValueError('Algorithm must be either "boston" or "gs"')

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
            curr_utilities = calculate_utilities_from_prob_individual(student=student,
                                                           num_schools=num_schools,
                                                           probabilities=curr_probabilities,
                                                           profiles=profiles)

            # print("For print", student)
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
                new_utilities = calculate_utilities_from_prob_individual(student=student,
                                                              num_schools=num_schools,
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

    p1 = k_gs_algorithm_prob(num_students = num_students, num_schools = num_schools, preferences = preferences,
                          capacities = capacities, k = k)
    print(p1)

    p2 = k_gs_algorithm_prob_2(num_students=num_students, num_schools=num_schools, preferences=preferences,
                          capacities=capacities, k=k)
    print(p2)

    p3 = k_gs_algorithm_prob_individual(num_students=num_students, num_schools=num_schools, preferences=preferences,
                               capacities=capacities, k=k, student=3)
    print(p3)

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
    # utilities = calculate_utilities_from_prob(num_students=num_students,
    #                                           num_schools=num_schools,
    #                                           probabilities=probabilities,
    #                                           profiles=profiles)
    #
    # print("true utilities", utilities, sep='\n')


