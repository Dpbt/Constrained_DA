import numpy as np
from typing import Dict, List, Set, Tuple
import itertools


def chinese_parallel_mechanism(num_students: int, num_schools: int,
                               preferences: np.ndarray, capacities: np.ndarray,
                               k: int, school_preferences: list) -> Dict[int, List[int]]:
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
        # curr_prefs = np.zeros(shape=(num_students, k))
        curr_prefs = np.full((num_students, k), -1)
        # print(1, "round_num", round_num, "remaining_students", remaining_students,
        #       "remaining_capacities", remaining_capacities)
        for student in remaining_students:
            start = round_num * k
            end = (round_num + 1) * k
            # Берем текущий блок предпочтений или остаток, если блок меньше k
            block = preferences[student][start:end]
            # print("student", student, "block", block)
            # Если блок меньше k, дополняем -1 (no preference)
            if len(block) < k:
                block = np.append(block, [-1] * (k - len(block)))
            curr_prefs[student] = block

        # print("curr_prefs", curr_prefs)

        # Запуск ограниченного алгоритма Гейла-Шепли для текущего раунда
        round_assignments, unmatched = k_gs_algorithm(
            num_students=num_students,
            num_schools=num_schools,
            preferences=curr_prefs,
            capacities=remaining_capacities,
            k=k,
            school_preferences=school_preferences
        )

        # print("round_assignments", round_assignments, "unmatched", unmatched)

        # print("round_assignments", round_assignments)

        # Обновляем финальные назначения и оставшиеся места
        for school in round_assignments:
            final_assignments[school].extend(round_assignments[school])
            remaining_capacities[school] -= len(round_assignments[school])

            for student in round_assignments[school]:
                final_assignments_students.add(student)
                if student in remaining_students:
                    remaining_students.remove(student)

        # Обновляем множество оставшихся студентов
        for student in unmatched:
            if student not in final_assignments_students:
                remaining_students.add(student)

        # Переходим к следующему раунду
        round_num += 1
        # print()

    return final_assignments


def k_gs_algorithm(num_students: int, num_schools: int, preferences: np.ndarray, capacities: np.ndarray, k: int,
                   school_preferences: list) -> Tuple[Dict[int, List[int]], Set[int]]:
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

            # print("school", school, "current_applicants", current_applicants)

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
                # students_to_assign = np.random.choice(all_current_applicants, size=capacities[school], replace=False)

                sorted_students = sorted(all_current_applicants, key=lambda x: school_preferences[x])

                # print("k_gs", school, all_current_applicants, sorted_students, school_preferences)

                students_to_assign = sorted_students[:capacities[school]]

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


def generate_possible_preferences(num_schools: int, k: int) -> np.ndarray:
    """
    Генерирует все возможные предпочтения, где:
    - сначала выбираются k школ в порядке возрастания
    - затем следующие k школ из оставшихся (также в порядке возрастания)
    - и так далее, пока не будут использованы все школы

    Параметры:
    num_schools -- общее количество школ (нумерация с 0)
    k -- размер блока для выбора школ

    Возвращает:
    Массив NumPy со всеми возможными комбинациями предпочтений
    """
    schools = np.arange(num_schools)
    all_combinations = []

    def generate_blocks(remaining_schools, current_pref):
        if len(remaining_schools) == 0:
            all_combinations.append(current_pref)
            return

        # Выбираем k школ (или меньше, если осталось меньше k)
        n = min(k, len(remaining_schools))
        for selected in itertools.combinations(remaining_schools, n):
            # Сортируем выбранные школы в порядке возрастания
            sorted_selected = sorted(selected)  # Изменено на возрастание
            new_remaining = [s for s in remaining_schools if s not in selected]
            generate_blocks(new_remaining, current_pref + list(sorted_selected))

    generate_blocks(list(schools), [])
    return np.array(all_combinations, dtype=np.int32)


def generate_paired_preferences(possible_preferences):
    """
    Генерирует массивы, где:
    - первый и второй элементы = одна строка из possible_preferences
    - третий и четвертый элементы = другая строка из possible_preferences

    Возвращает массив shape (n, 4, num_schools), где n - число комбинаций
    """
    n_prefs = len(possible_preferences)
    paired = []

    # Генерируем все возможные пары (с повторами)
    for i in range(n_prefs):
        for j in range(n_prefs):
            # Создаем массив 4 x num_schools
            new_array = np.vstack([
                possible_preferences[i],
                possible_preferences[i],
                possible_preferences[j],
                possible_preferences[j]
            ])
            paired.append(new_array)

    return np.array(paired)


def generate_preferences(utils: list, num_students: int, k: int) -> list:
    """
    Генерирует профили предпочтений в формате кортежей с массивами NumPy.
    """
    num_schools = len(utils[0]) if num_students > 0 else 0

    # Группируем учеников по их utils
    groups = {}
    for i in range(num_students):
        key = tuple(utils[i])
        groups.setdefault(key, []).append(i)

    # Генерируем возможные предпочтения для каждой группы
    prefs_per_group = {
        u: generate_possible_preferences(num_schools, k) for u in groups.keys()
    }

    # Создаём комбинации для уникальных групп
    group_keys = list(groups.keys())
    product_inputs = [prefs_per_group[key] for key in group_keys]

    # Генерируем итоговые профили
    profile_prefs = []
    for combo in itertools.product(*product_inputs):
        profile = [None] * num_students
        for group_idx, pref_array in enumerate(combo):
            student_indices = groups[group_keys[group_idx]]
            for student_idx in student_indices:
                profile[student_idx] = pref_array
        profile_prefs.append(tuple(np.array(p, dtype=np.int32) for p in profile))

    return profile_prefs


def find_nash_equilibrium(results: list):
    utility_dict = {}
    for pref, utils in results:
        p1_tuple = tuple(int(x) for x in pref[0])
        p2_tuple = tuple(int(x) for x in pref[1])
        utility_dict[(p1_tuple, p2_tuple)] = (float(utils[0]), float(utils[1]))

    # Получаем уникальные стратегии
    all_p1 = list({tuple(int(x) for x in pref[0]) for pref, _ in results})
    all_p2 = list({tuple(int(x) for x in pref[1]) for pref, _ in results})

    # print("all_p1", all_p1)

    # Функция для проверки, является ли пара (p1, p2) равновесием Нэша
    def is_nash_equilibrium(p1, p2):
        u1_p1p2, u2_p1p2 = utility_dict[(p1, p2)]
        # print("u1_p1p2", u1_p1p2, "u2_p1p2", u2_p1p2)

        # Проверяем, может ли первый игрок улучшить u1, меняя p1 (при фиксированном p2)
        for p1_candidate in all_p1:
            if p1_candidate == p1:
                continue
            u1_new, u2_new = utility_dict[(p1_candidate, p2)]
            if u1_new > u1_p1p2:
                print("Not Nash 1", (p1, p2), p1_candidate, (u1_p1p2, u2_p1p2), (u1_new, u2_new))
                return False

        # Проверяем, может ли второй игрок улучшить u2, меняя p2 (при фиксированном p1)
        for p2_candidate in all_p2:
            if p2_candidate == p2:
                continue
            u1_new, u2_new = utility_dict[(p1, p2_candidate)]
            if u2_new > u2_p1p2:
                print("Not Nash 2", (p1, p2), p2_candidate, (u1_p1p2, u2_p1p2), (u1_new, u2_new))
                return False

        return True

    # Ищем все равновесия Нэша
    nash_equilibria = []
    for p1 in all_p1:
        for p2 in all_p2:
            if (p1, p2) in utility_dict and is_nash_equilibrium(p1, p2):
                nash_equilibria.append((p1, p2, utility_dict[(p1, p2)]))

    print("Найдены следующие равновесия Нэша:")
    for p1, p2, (u1, u2) in nash_equilibria:
        print(f"Player 1: {p1}, Player 2: {p2}, Utilities: {u1:.2f}, {u2:.2f}")


def find_nash_equilibrium_general(results: list):
    # Создаем словарь полезностей
    utility_dict = {}
    for pref, utils in results:
        strategy_profile = tuple(
            tuple(int(x) for x in player_strategy) for player_strategy in pref
        )
        utilities = tuple(float(u) for u in utils)
        utility_dict[strategy_profile] = utilities

    # Определяем количество игроков
    num_players = len(next(iter(utility_dict.keys())))

    # Собираем уникальные стратегии для каждого игрока
    players_strategies = []
    for player_idx in range(num_players):
        strategies = {tuple(int(x) for x in pref[player_idx]) for pref, _ in results}
        players_strategies.append(list(strategies))

    # Функция проверки равновесия Нэша
    def is_nash_equilibrium(strategy_profile):
        current_utilities = utility_dict[strategy_profile]

        for player_idx in range(num_players):
            # Сравниваем с альтернативными стратегиями текущего игрока
            for candidate_strategy in players_strategies[player_idx]:
                if candidate_strategy == strategy_profile[player_idx]:
                    continue

                # Создаем новый профиль стратегий
                new_profile = list(strategy_profile)
                new_profile[player_idx] = candidate_strategy
                new_profile = tuple(new_profile)

                # Проверяем улучшение полезности
                if new_profile not in utility_dict:
                    continue
                if utility_dict[new_profile][player_idx] > current_utilities[player_idx]:
                    return False
                # if (
                #     utility_dict[new_profile][player_idx]
                #     > current_utilities[player_idx]
                # ):
                #     return False
        return True

    # Генерируем все возможные профили стратегий
    nash_equilibria = []
    for strategy_profile in itertools.product(*players_strategies):
        if strategy_profile in utility_dict and is_nash_equilibrium(strategy_profile):
            nash_equilibria.append((strategy_profile, utility_dict[strategy_profile]))

    # Выводим результаты
    print("Найдены следующие равновесия Нэша:")
    for profile, utils in nash_equilibria:
        players_str = ", ".join(
            f"Player {i+1}: {strat}" for i, strat in enumerate(profile)
        )
        utils_str = ", ".join(f"{u:.2f}" for u in utils)
        print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


def find_nash_equilibrium_grouped(results: list, player_utils: list):
    """
    player_utils: список utility-профилей для каждого игрока [[79,16,4,1], [79,16,4,1], ...]
    """
    # Создаем словарь полезностей для профилей стратегий
    utility_dict = {}
    for pref, utils in results:
        strategy_profile = tuple(
            tuple(int(x) for x in player_strategy) for player_strategy in pref
        )
        utilities = tuple(float(u) for u in utils)
        utility_dict[strategy_profile] = utilities

    # Группируем игроков по их utility-профилям
    utils_groups = {}
    for player_idx, utils in enumerate(player_utils):
        key = tuple(utils)
        utils_groups.setdefault(key, []).append(player_idx)

    # Собираем уникальные стратегии для каждого игрока
    players_strategies = []
    num_players = len(next(iter(utility_dict.keys())))
    for player_idx in range(num_players):
        strategies = {tuple(int(x) for x in pref[player_idx]) for pref, _ in results}
        players_strategies.append(list(strategies))

    # Функция проверки равновесия Нэша с групповыми отклонениями
    def is_nash_equilibrium(strategy_profile):
        current_utilities = utility_dict[strategy_profile]

        # Проверяем для каждой группы игроков
        for group_indices in utils_groups.values():
            # Текущая стратегия группы (должна быть одинаковой у всех)
            group_strategy = strategy_profile[group_indices[0]]

            # Проверяем, что все в группе используют одинаковую стратегию
            if any(strategy_profile[i] != group_strategy for i in group_indices):
                continue  # Пропускаем если стратегии разные

            # Перебираем возможные альтернативные стратегии для группы
            for candidate_strategy in players_strategies[group_indices[0]]:
                if candidate_strategy == group_strategy:
                    continue  # Пропускаем текущую стратегию

                # Создаем новый профиль стратегий
                new_profile = list(strategy_profile)
                for i in group_indices:
                    new_profile[i] = candidate_strategy
                new_profile = tuple(new_profile)

                # Проверяем существование профиля
                if new_profile not in utility_dict:
                    continue

                # Проверяем улучшение для ВСЕХ игроков группы
                all_improved = all(
                    utility_dict[new_profile][i] > current_utilities[i]
                    for i in group_indices
                )

                if all_improved:
                    return False

        return True

    # Генерируем профили стратегий с учетом групповых ограничений
    nash_equilibria = []
    for strategy_profile in itertools.product(*players_strategies):
        if strategy_profile not in utility_dict:
            continue

        # Проверяем чтобы внутри групп стратегии совпадали
        valid_profile = True
        for group_indices in utils_groups.values():
            group_strategies = [strategy_profile[i] for i in group_indices]
            if len(set(group_strategies)) != 1:
                valid_profile = False
                break

        if valid_profile and is_nash_equilibrium(strategy_profile):
            nash_equilibria.append((strategy_profile, utility_dict[strategy_profile]))

    # Выводим результаты
    print("Найдены следующие равновесия Нэша:")
    for profile, utils in nash_equilibria:
        players_str = ", ".join(
            f"Player {i+1}: {strat}" for i, strat in enumerate(profile)
        )
        utils_str = ", ".join(f"{u:.2f}" for u in utils)
        print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


if __name__ == '__main__':
    num_schools = 5
    num_students = 5
    k = 3
    capacities = np.array([1, 1, 1, 1, 1])

    # profiles = [[54, 23, 15, 8], [54, 23, 15, 8], [38, 32, 30, 0], [38, 32, 30, 0]]
    # profiles = [[90, 6, 4, 0], [90, 8, 2, 0], [90, 9, 1, 0], [90, 10, 0, 0]]
    profiles = [[90, 8, 2, 0], [90, 8, 2, 0], [90, 9, 1, 0], [90, 10, 0, 0]]
    # profiles = [[50, 40, 5, 5], [51, 45, 4, 0], [52, 45, 3, 0], [53, 45, 2, 0]]
    # profiles = [[79, 16, 4, 1], [79, 16, 4, 1], [79, 10, 10, 1], [79, 10, 10, 1]]
    # profiles = [[79, 16, 3, 2], [79, 16, 3, 2], [79, 16, 5, 0], [79, 16, 5, 0]]
    # profiles = [[79, 16, 3, 2], [79, 16, 3, 2], [79, 16, 5, 0], [79, 16, 5, 0]]
    profiles = [[50, 40, 9, 1], [50, 40, 8, 2], [50, 40, 7, 3], [50, 30, 6, 4]]
    profiles = [[50, 40, 5, 3, 2], [50, 40, 5, 3, 2], [50, 40, 5, 3, 2], [50, 40, 5, 3, 2], [50, 40, 5, 4, 1]]

    student_rank_default = [i for i in range(num_students)]
    student_ranks = list(itertools.permutations(student_rank_default))

    # possible_preferences = generate_possible_preferences(num_schools, k)
    # paired_prefs = generate_paired_preferences(possible_preferences)
    # profile_prefs = list(itertools.product(possible_preferences, repeat=num_students))

    profile_prefs = generate_preferences(profiles, num_students, k)
    print(len(profile_prefs))
    # for pr in profile_prefs:
    #     print(pr)
    # print(profile_prefs)

    results = []

    algorithm = "chinese_parallel_mechanism"  # или "k_gs_algorithm"
    # algorithm = "k_gs_algorithm"

    for i, profile_pref in enumerate(profile_prefs):
        # if i % 1000 == 0:
        #     print(i)
        pref_utils = np.zeros(num_students)
        for student_rank in student_ranks:
            if algorithm == "chinese_parallel_mechanism":
                # Используем китайский параллельный механизм
                school_assignments = chinese_parallel_mechanism(
                    num_students=num_students,
                    num_schools=num_schools,
                    preferences=np.array(profile_pref),
                    capacities=capacities,
                    k=k,
                    school_preferences=student_rank
                )

                # print(school_assignments)

            elif algorithm == "k_gs_algorithm":
                profile_pref = np.array(profile_pref)
                paired_pref = profile_pref[:, :2]
                # Используем алгоритм Гейла-Шепли
                school_assignments, unassigned = k_gs_algorithm(
                    num_students=num_students,
                    num_schools=num_schools,
                    preferences=np.array(profile_pref),
                    capacities=capacities,
                    k=k,
                    school_preferences=student_rank
                )

            # print("paired pref", paired_pref, "school_assignments", school_assignments, unassigned)

            student_assignments = {}
            for school, students in school_assignments.items():
                for student in students:
                    student_assignments[student] = school

            # print(profile_pref)
            # print(student_rank)
            # print(student_assignments)

            if algorithm == "chinese_parallel_mechanism":
                curr_utils = np.array([profiles[student][student_assignments[student]] for student in range(num_students)])

            elif algorithm == "k_gs_algorithm":
                curr_utils = np.array([profiles[student][student_assignments[student]] if student not in unassigned
                                       else 0 for student in range(num_students)])

            pref_utils += curr_utils

            # print(student_assignments, unassigned, curr_utils)

        pref_utils /= len(student_ranks)
        # results.append((paired_pref[[0, 2]], pref_utils[[0, 2]]))
        results.append((profile_pref, pref_utils))

    find_nash_equilibrium_grouped(results, profiles)
    print()

    # algorithm = "chinese_parallel_mechanism"  # или "k_gs_algorithm"
    #
    # profile_pref = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 3, 2]])
    # pref_utils = np.zeros(num_students)
    #
    # for student_rank in student_ranks:
    #     if algorithm == "chinese_parallel_mechanism":
    #         # Используем китайский параллельный механизм
    #         school_assignments = chinese_parallel_mechanism(
    #             num_students=num_students,
    #             num_schools=num_schools,
    #             preferences=np.array(profile_pref),
    #             capacities=np.array([1, 1, 1, 1]),
    #             k=k,
    #             school_preferences=student_rank,
    #         )
    #
    #         # print(school_assignments)
    #
    #     elif algorithm == "k_gs_algorithm":
    #         profile_pref = np.array(profile_pref)
    #         paired_pref = profile_pref[:, :2]
    #         # Используем алгоритм Гейла-Шепли
    #         school_assignments, unassigned = k_gs_algorithm(
    #             num_students=num_students,
    #             num_schools=num_schools,
    #             preferences=np.array(profile_pref),
    #             capacities=np.array([1, 1, 1, 1]),
    #             k=k,
    #             school_preferences=student_rank,
    #         )
    #
    #     # print("paired pref", paired_pref, "school_assignments", school_assignments, unassigned)
    #
    #     student_assignments = {}
    #     for school, students in school_assignments.items():
    #         for student in students:
    #             student_assignments[student] = school
    #
    #     # print(profile_pref)
    #     # print(student_rank)
    #     # print(student_assignments)
    #
    #     if algorithm == "chinese_parallel_mechanism":
    #         curr_utils = np.array(
    #             [
    #                 profiles[student][student_assignments[student]]
    #                 for student in range(num_students)
    #             ]
    #         )
    #
    #     elif algorithm == "k_gs_algorithm":
    #         curr_utils = np.array(
    #             [
    #                 (
    #                     profiles[student][student_assignments[student]]
    #                     if student not in unassigned
    #                     else 0
    #                 )
    #                 for student in range(num_students)
    #             ]
    #         )
    #
    #     pref_utils += curr_utils
    #
    #     # print(student_assignments, unassigned, curr_utils)
    #
    # pref_utils /= len(student_ranks)
    #
    # print(pref_utils)
