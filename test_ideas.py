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


if __name__ == '__main__':
    preferences = np.array([0, 1])
    print(generate_possible_manipulations(num_schools=3, preferences=preferences, k=2))
