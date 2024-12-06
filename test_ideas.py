import numpy as np
from utils import generate_subsets

# Пример двумерного массива размером n x m
n = 4  # количество строк
m = 5  # количество столбцов
x = np.random.rand(n, m)  # создание массива с случайными числами

# Значение k (максимальный индекс для суммирования)
k = 3

# Вычисление суммы элементов от 0 до k для каждой строки
sums = np.sum(x[:, :k+1], axis=1)

# Вычитание суммы из 1
result = 1 - sums

print("Массив x:")
print(x)
print(f"\nСуммы элементов от 0 до {k} для каждой строки:")
print(sums)
print(f"\nРезультат (1 - сумма):")
print(result)

print(generate_subsets([1, 2, 3 ,5], 4))


a = np.array([1, 2, 3, 4, 5])
mask = [0, 1, 4]
mask = []
print(a[mask])