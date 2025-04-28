import numpy as np
import itertools
from algorithm_modified import k_gs_algorithm, chinese_parallel_mechanism
from utils import (AlgorithmEnum)


# utils.py
def generate_possible_preferences_k_gs(num_schools: int, k: int) -> np.ndarray:
    """
    Генерирует все возможные комбинации из k школ в порядке возрастания.

    Параметры:
    num_schools -- общее количество школ (нумерация с 0)
    k -- количество школ в предпочтении

    Возвращает:
    Массив NumPy формы (C(num_schools, k), k) с отсортированными комбинациями
    """
    schools = np.arange(num_schools)
    combinations = list(itertools.combinations(schools, k))
    return np.array(combinations, dtype=np.int32)


# utils.py
def generate_possible_preferences_chinese(num_schools: int, k: int) -> np.ndarray:
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


# utils.py
def generate_symmetric_preferences(profiles: list,
                                   num_students: int,
                                   k: int,
                                   get_possible_preferences_func=generate_possible_preferences_chinese) -> list:
    """
    Генерирует профили предпочтений в формате кортежей с массивами NumPy.
    """
    num_schools = len(profiles[0]) if num_students > 0 else 0

    # Группируем учеников по их utils
    groups = {}
    for i in range(num_students):
        key = tuple(profiles[i])
        groups.setdefault(key, []).append(i)

    # Генерируем возможные предпочтения для каждой группы
    possible_preferences = get_possible_preferences_func(num_schools, k)
    prefs_per_group = {u: possible_preferences for u in groups.keys()}

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

        profile_prefs.append(np.array(profile))

    return profile_prefs


# utils.py ?
def find_nash_equilibrium(results: list):
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
                if (
                        utility_dict[new_profile][player_idx]
                        > current_utilities[player_idx]
                ):
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
            f"Player {i + 1}: {strat}" for i, strat in enumerate(profile)
        )
        utils_str = ", ".join(f"{u:.2f}" for u in utils)
        print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


# utils.py ?
def find_symmetric_nash_equilibrium(results: list, player_utils: list):
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
    if not nash_equilibria:
        print("Не найдены равновесия Нэша.")
    else:
        print("Найдены следующие равновесия Нэша:")
        for profile, utils in nash_equilibria:
            players_str = ", ".join(
                f"Player {i + 1}: {strat}" for i, strat in enumerate(profile)
            )
            utils_str = ", ".join(f"{u:.2f}" for u in utils)
            print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


def all_preferences_test(
        num_schools: int,
        num_students: int,
        k: int,
        capacities: np.ndarray,
        profiles: list,
        algorithm: AlgorithmEnum = AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
):
    get_possible_preferences_func, algorithm_func = None, None

    if algorithm == AlgorithmEnum.CHINESE_PARALLEL_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_chinese
        algorithm_func = chinese_parallel_mechanism

    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_k_gs
        algorithm_func = k_gs_algorithm

    student_rank_default = [i for i in range(num_students)]
    student_ranks = list(itertools.permutations(student_rank_default))

    profile_prefs = generate_symmetric_preferences(profiles=profiles,
                                                   num_students=num_students,
                                                   k=k,
                                                   get_possible_preferences_func=get_possible_preferences_func)

    results = []
    school_assignments = None

    for i, profile_pref in enumerate(profile_prefs):
        pref_utils = np.zeros(num_students)

        for student_rank in student_ranks:
            school_assignments, unassigned = algorithm_func(
                num_students=num_students,
                num_schools=num_schools,
                preferences=profile_pref,
                capacities=capacities,
                k=k,
                school_preferences=student_rank,
            )

            student_assignments = {}
            for school, students in school_assignments.items():
                for student in students:
                    student_assignments[student] = school

            curr_utils = np.array(
                [(profiles[student][student_assignments[student]]
                  if student not in unassigned else 0)
                 for student in range(num_students)])

            pref_utils += curr_utils

        pref_utils /= len(student_ranks)
        results.append((profile_pref, pref_utils))

    return results


if __name__ == "__main__":
    num_schools = 4
    num_students = 4
    k = 1
    capacities = np.array([1, 1, 1, 1, 1])

    # profiles = [[54, 23, 15, 8], [54, 23, 15, 8], [38, 32, 30, 0], [38, 32, 30, 0]]

    # profiles = [
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 4, 1],
    # ]

    profiles = [
        [90, 6, 4, 0],
        [90, 8, 2, 0],
        [90, 9, 1, 0],
        [90, 10, 0, 0],
    ]

    results = all_preferences_test(
        num_schools=num_schools,
        num_students=num_students,
        k=k,
        capacities=capacities,
        profiles=profiles,
        # algorithm=AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
        algorithm=AlgorithmEnum.K_GS_MECHANISM,
    )

    find_symmetric_nash_equilibrium(results, profiles)
