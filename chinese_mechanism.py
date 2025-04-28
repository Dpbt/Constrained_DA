import numpy as np
import itertools
from algorithm_modified import k_gs_algorithm, chinese_parallel_mechanism
from utils import (AlgorithmEnum)
from typing import Callable


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
def generate_symmetric_preferences(num_students: int,
                                   num_schools: int,
                                   k: int,
                                   profiles: np.ndarray,
                                   get_possible_preferences_func: Callable[[int, int], np.ndarray]
                                   = generate_possible_preferences_chinese) -> list:
    """
    Генерирует профили предпочтений в формате кортежей с массивами NumPy.
    """
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


# utils.py
def find_nash_equilibrium(
        results: list[tuple[np.ndarray, np.ndarray]],
        profiles: np.ndarray = None,
        symmetric: bool = False
) -> list[tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]]:
    """
    Finds Nash equilibria (symmetric or all) in strategic games.

    Parameters:
    results -- list of tuples (strategy_profile, utilities)
    profiles -- np.ndarray of players utility profiles (for symmetric=True)
    symmetric -- flag to search only for symmetric equilibria

    Returns:
    List of tuples (strategy_profile, utilities)
    """
    utility_dict = {}
    # Create utility dictionary
    for pref, utils in results:
        strategy_profile = tuple(
            tuple(int(x) for x in player_strategy) for player_strategy in pref
        )
        utilities = tuple(float(u) for u in utils)
        utility_dict[strategy_profile] = utilities

    # Determine number of players
    num_players = len(next(iter(utility_dict.keys())))
    players_strategies = []

    # Collect unique strategies for each player
    for player_idx in range(num_players):
        strategies = {tuple(int(x) for x in pref[player_idx]) for pref, _ in results}
        players_strategies.append(list(strategies))

    # Player grouping logic (for symmetric case only)
    utils_groups = {}
    if symmetric:
        if profiles is None:
            raise ValueError("profiles parameter is required when symmetric=True")

        for player_idx, prefs in enumerate(profiles):
            key = tuple(int(x) for x in prefs)
            utils_groups.setdefault(key, []).append(player_idx)

    def is_nash_equilibrium(strategy_profile):
        """Check Nash equilibrium conditions"""
        current_utilities = utility_dict[strategy_profile]

        if symmetric:
            # Check group deviations
            for group_indices in utils_groups.values():
                group_strategy = strategy_profile[group_indices[0]]

                if any(strategy_profile[i] != group_strategy for i in group_indices):
                    continue

                for candidate_strategy in players_strategies[group_indices[0]]:
                    if candidate_strategy == group_strategy:
                        continue

                    # Create new strategy profile
                    new_profile = list(strategy_profile)
                    for i in group_indices:
                        new_profile[i] = candidate_strategy
                    new_profile = tuple(new_profile)

                    if new_profile not in utility_dict:
                        continue

                    if all(utility_dict[new_profile][i] > current_utilities[i] for i in group_indices):
                        return False
        else:
            # Standard check for all players
            for player_idx in range(num_players):
                for candidate_strategy in players_strategies[player_idx]:
                    if candidate_strategy == strategy_profile[player_idx]:
                        continue

                    new_profile = list(strategy_profile)
                    new_profile[player_idx] = candidate_strategy
                    new_profile = tuple(new_profile)

                    if new_profile not in utility_dict:
                        continue

                    if utility_dict[new_profile][player_idx] > current_utilities[player_idx]:
                        return False
        return True

    # Generate strategy profiles considering symmetry
    nash_equilibria = []
    for strategy_profile in itertools.product(*players_strategies):
        if strategy_profile not in utility_dict:
            continue

        # Verify strategy consistency within groups
        valid_profile = True
        if symmetric:
            for group_indices in utils_groups.values():
                if len({strategy_profile[i] for i in group_indices}) != 1:
                    valid_profile = False
                    break

        if valid_profile and is_nash_equilibrium(strategy_profile):
            nash_equilibria.append((strategy_profile, utility_dict[strategy_profile]))

    # Format output
    if not nash_equilibria:
        print("No Nash equilibria found.")
    else:
        print(f"Found {len(nash_equilibria)} Nash equilibria ({'symmetric' if symmetric else 'all types'}):")
        for profile, utils in nash_equilibria:
            players_str = ", ".join(f"Player {i + 1}: {strat}" for i, strat in enumerate(profile))
            utils_str = ", ".join(f"{u:.2f}" for u in utils)
            print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria


def all_preferences_test(
        num_schools: int,
        num_students: int,
        k: int,
        capacities: np.ndarray,
        profiles: np.ndarray,
        algorithm: AlgorithmEnum = AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
):
    get_possible_preferences_func, algorithm_func = None, None

    if algorithm == AlgorithmEnum.CHINESE_PARALLEL_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_chinese
        algorithm_func = chinese_parallel_mechanism

    elif algorithm == AlgorithmEnum.K_GS_MECHANISM:
        get_possible_preferences_func = generate_possible_preferences_k_gs
        algorithm_func = k_gs_algorithm

    school_preferences_default = [i for i in range(num_students)]
    school_preferences = list(itertools.permutations(school_preferences_default))

    all_symmetric_preferences = generate_symmetric_preferences(num_students=num_students,
                                                               k=k,
                                                               num_schools=num_schools,
                                                               profiles=profiles,
                                                               get_possible_preferences_func=get_possible_preferences_func)

    results = []

    for i, preferences in enumerate(all_symmetric_preferences):
        preferences_utils = np.zeros(num_students)

        for school_preference in school_preferences:
            school_assignments, unassigned = algorithm_func(
                num_students=num_students,
                num_schools=num_schools,
                preferences=preferences,
                capacities=capacities,
                k=k,
                school_preferences=school_preference,
            )

            student_assignments = {}
            for school, students in school_assignments.items():
                for student in students:
                    student_assignments[student] = school

            curr_utils = np.array(
                [(profiles[student][student_assignments[student]]
                  if student not in unassigned else 0)
                 for student in range(num_students)])

            preferences_utils += curr_utils

        preferences_utils /= len(school_preferences)
        results.append((preferences, preferences_utils))

    return results


def get_type_structure(obj) -> str:
    if isinstance(obj, list):
        if not obj: return "List[Any]"
        return f"List[{get_type_structure(obj[0])}]"

    elif isinstance(obj, tuple):
        types = ", ".join(get_type_structure(x) for x in obj)
        return f"Tuple[{types}]"

    elif isinstance(obj, dict):
        if not obj: return "Dict[Any, Any]"
        key_type = get_type_structure(next(iter(obj.keys())))
        val_type = get_type_structure(next(iter(obj.values())))
        return f"Dict[{key_type}, {val_type}]"

    return type(obj).__name__


if __name__ == "__main__":
    # num_schools = 4
    # num_students = 4
    # k = 2
    # capacities = np.array([1, 1, 1, 1, 1])

    profiles = np.array([[54, 23, 15, 8], [54, 23, 15, 8], [38, 32, 30, 0], [38, 32, 30, 0]])

    # profiles = np.array([
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 3, 2],
    #     [50, 40, 5, 4, 1],
    # ])

    # profiles = np.array([
    #     [90, 6, 4, 0],
    #     [90, 8, 2, 0],
    #     [90, 9, 1, 0],
    #     [90, 10, 0, 0],
    # ])

    num_schools = profiles.shape[1]
    num_students = profiles.shape[0]
    k = 2
    capacities = np.array([1 for _ in range(num_schools)])

    results = all_preferences_test(
        num_schools=num_schools,
        num_students=num_students,
        k=k,
        capacities=capacities,
        profiles=profiles,
        # algorithm=AlgorithmEnum.CHINESE_PARALLEL_MECHANISM,
        algorithm=AlgorithmEnum.K_GS_MECHANISM,
    )

    find_nash_equilibrium(results=results, profiles=profiles, symmetric=True)
