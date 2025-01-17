import numpy as np
import pandas as pd
import time
from numba import jit, njit


import numpy as np
from numba import njit


def generate_possible_manipulations(num_schools: int, preferences: np.ndarray, k: int) -> np.ndarray:
    if len(preferences) != k or not np.all(np.diff(preferences) > 0):
        raise ValueError("Preferences must be a sorted array of length k.")

    preferences_set = set(preferences)  # Преобразуем в set для быстрого доступа
    manipulations = []

    for i in range(k):
        current_preference = preferences[i]

        # Проверяем школы, которые выше текущего
        for next_school in range(current_preference + 1, num_schools):
            if next_school not in preferences_set:
                new_preferences = preferences.copy()
                new_preferences[i] = next_school
                manipulations.append(np.sort(new_preferences))
                break

    # Убираем дублирование
    unique_manipulations = np.unique(np.array(manipulations), axis=0)

    return unique_manipulations


import itertools


def generate_dict_list(*param_lists):
    # Получаем имена параметров из переданных аргументов
    param_names = [f'param{i + 1}' for i in range(len(param_lists))]

    # Используем itertools.product для получения всех комбинаций
    combinations = itertools.product(*param_lists)

    # Создаем список словарей
    result = [dict(zip(param_names, combo)) for combo in combinations]

    return result


if __name__ == '__main__':
    # preferences = np.array([0, 1])
    # # print(generate_possible_manipulations(num_schools=3, preferences=preferences, k=2))
    #
    #
    # data = {
    #     "Column A": 1,
    #     "Column B": 4,
    #     "Column C": 7
    # }
    #
    # df = pd.DataFrame(data, index=[0])
    # df2 = pd.DataFrame(data, index=[0])
    #
    # experiment_results = pd.concat([df, df2], ignore_index=True)
    #
    # # print(experiment_results)
    # utilities = np.array([10, 20, 30, 40])
    # mask = np.zeros(4, dtype=bool)
    # mask[np.array([1, 3])] = True
    #
    # # Получаем массив элементов, не входящих в маску
    # result = utilities[mask]
    # result = utilities[~np.array([1, 3])]
    # print(result)

    # Пример использования
    param1 = [1, 2]
    param2 = ['a', 'b']
    param3 = [True, False]

    result = generate_dict_list(param1, param2, param3)
    for item in result:
        print(item)