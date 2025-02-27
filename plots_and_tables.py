import pandas as pd
import numpy as np
import matplotlib
import os
import itertools
import json


def utility_table_students_schools_k(file_path, output_file, num_best=1, only_gs=False, filters=None):
    # Чтение данных из файла
    df = pd.read_csv(file_path)

    if only_gs:
        # Фильтрация данных только для алгоритма GS
        df = df[df['algorithm'] == 'gs']

    # Применение дополнительных фильтров
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                if column == 'num_manipulations':
                    mask = pd.Series(False, index=df.index)
                    for value in values:
                        mask |= (df['num_manipulations'] == round(value * df['num_schools']))
                    df = df[mask]
                else:
                    df = df[df[column].isin(values)]

    # Сортировка данных по эксперименту и рейтингу полезности
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Выбор num_best лучших результатов для каждого эксперимента
    best_utility_df = df.groupby('experiment_number').head(num_best)

    # Группировка данных по числу учеников и школ, вычисление среднего k_to_schools_ratio
    pivot_table = best_utility_df.groupby(['num_students', 'num_schools'])['k_to_schools_ratio'].mean().unstack()

    # Сортировка индексов и столбцов
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Заполнение NaN значений прочерками
    pivot_table = pivot_table.fillna('-')

    output_file = output_file + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"
    # Сохранение таблицы в указанный файл
    pivot_table.to_csv(output_file, index=True)

    print(pivot_table)

    print(f"Таблица успешно сохранена в {output_file}")


def create_unassigned_students_table(file_path, output_file, num_best=1, only_gs=False, target="all", filters=None):
    # Чтение данных из файла
    df = pd.read_csv(file_path)

    target_dict = {
        "all": "average_percentage_unassigned_students",
        "fair": "average_percentage_unassigned_fair_students",
        "manipulators": "average_percentage_unassigned_manipulator_students",
    }

    target = target_dict[target]

    if only_gs:
        # Фильтрация данных только для алгоритма GS
        df = df[df['algorithm'] == 'gs']

    # Применение дополнительных фильтров
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                if column == 'num_manipulations':
                    mask = pd.Series(False, index=df.index)
                    for value in values:
                        mask |= (df['num_manipulations'] == round(value * df['num_schools']))
                    df = df[mask]
                else:
                    df = df[df[column].isin(values)]

    # Сортировка данных по эксперименту и рейтингу полезности
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Выбор num_best лучших результатов для каждого эксперимента
    best_utility_df = df.groupby('experiment_number').head(num_best)

    pivot_table = best_utility_df.groupby(['num_students', 'num_schools'])[target].mean().unstack()

    # Сортировка индексов и столбцов
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Заполнение NaN значений прочерками
    pivot_table = pivot_table.fillna('-')

    output_file = output_file + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"
    # Сохранение таблицы в указанный файл
    pivot_table.to_csv(output_file, index=True)

    print(pivot_table)

    print(f"Таблица успешно сохранена в {output_file}")


def create_comprehensive_table(file_path, output_file, target, num_best=1, only_gs=False, filters=None,
                               groupby="num_students",
                               show_k_ratio=True, show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True):
    # Чтение данных из файла
    df = pd.read_csv(file_path)

    if only_gs:
        # Фильтрация данных только для алгоритма GS
        df = df[df['algorithm'] == 'gs']

    # Применение дополнительных фильтров
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                if column == 'num_manipulations':
                    mask = pd.Series(False, index=df.index)
                    for value in values:
                        mask |= (df['num_manipulations'] == round(value * df['num_schools']))
                    df = df[mask]
                else:
                    df = df[df[column].isin(values)]

    # Сортировка данных по эксперименту и рейтингу полезности
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Выбор num_best лучших результатов для каждого эксперимента
    best_utility_df = df.groupby('experiment_number').head(num_best)

    # Определение столбцов для агрегации
    agg_columns = ['k_to_schools_ratio'] if show_k_ratio else []
    if show_utility:
        agg_columns.append('average_utility')
        if show_fair:
            agg_columns.append('average_utility_fair_students')
        if show_manipulator:
            agg_columns.append('average_utility_manipulator_students')
    if show_unassigned:
        agg_columns.append('average_percentage_unassigned_students')
        if show_fair:
            agg_columns.append('average_percentage_unassigned_fair_students')
        if show_manipulator:
            agg_columns.append('average_percentage_unassigned_manipulator_students')

    # Группировка и агрегация данных
    groupby_columns = []
    if groupby:
        groupby_columns.append(groupby)
    grouped = best_utility_df.groupby([target] + groupby_columns)
    agg_data = grouped[agg_columns].mean()

    # Форматирование данных для вывода
    def format_cell(row, accuracy=2):
        cell_parts = []
        if show_k_ratio:
            cell_parts.append(f"K/S ratio: {row['k_to_schools_ratio'].iloc[0]:.{accuracy}f}")
        if show_utility:
            utility_parts = [f"U: {row['average_utility'].iloc[0]:.{accuracy}f}"]
            if show_fair:
                utility_parts.append(f"F: {row['average_utility_fair_students'].iloc[0]:.{accuracy}f}")
            if show_manipulator:
                utility_parts.append(f"M: {row['average_utility_manipulator_students'].iloc[0]:.{accuracy}f}")
            cell_parts.append(" (".join(utility_parts) + ")")
        if show_unassigned:
            unassigned_parts = [f"UA: {row['average_percentage_unassigned_students'].iloc[0]:.{accuracy}f}%"]
            if show_fair:
                unassigned_parts.append(f"F: {row['average_percentage_unassigned_fair_students'].iloc[0]:.{accuracy}f}%")
            if show_manipulator:
                unassigned_parts.append(f"M: {row['average_percentage_unassigned_manipulator_students'].iloc[0]:.{accuracy}f}%")
            cell_parts.append(" (".join(unassigned_parts) + ")")
        return " ".join(cell_parts)

    pivot_table = agg_data.groupby(level=[0, 1]).apply(format_cell).unstack(level=1)

    # Сортировка индексов и столбцов
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Заполнение NaN значений прочерками
    pivot_table = pivot_table.fillna('-')

    output_file = f"{output_file}_{num_best}_{'gs' if only_gs else 'all'}_{target}.csv"
    # Сохранение таблицы в указанный файл
    pivot_table.to_csv(output_file, index=True)

    print(pivot_table)
    print(f"Таблица успешно сохранена в {output_file}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Список колонок
    # df = pd.read_csv("./data_out/data_out_server_3_filtered.csv")
    # column_list = df.columns.tolist()
    # print(column_list)

    # 100 - 8, 500 - 5

    # Part 100-500

    utility_table_students_schools_k(file_path="./data_out/data_out_server_3.csv",
                                     output_file="./data_out/analysis/table_filtered_students_schools_k",
                                     num_best=1,
                                     only_gs=True,
                                     filters=None)

    # filters = {
    #     "num_students": [100, 200, 300, 400, 500],
    #     "num_schools": [5, 10, 20],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }
    #
    # filters = {
    #     "num_students": [400],
    #     "num_schools": [5],
    #     "epsilon": [0.005],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.75],
    # }
    #
    # create_comprehensive_table(file_path="./data_out/data_out_server_3.csv",
    #                            output_file="./data_out/analysis/table_test_1",
    #                            target="manipulators_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # filters = {
    #     "num_students": [400],
    #     "num_schools": [5],
    #     "epsilon": [0.005],
    #     "manipulators_ratio": [0.75],
    #     "num_manipulations_ratio": [0.5, 0.75, 1],
    # }
    #
    # create_comprehensive_table(file_path="./data_out/data_out_server_3.csv",
    #                            output_file="./data_out/analysis/table_test_1",
    #                            target="num_manipulations_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # filters = {
    #     "num_students": [400],
    #     "num_schools": [5],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.75],
    #     "num_manipulations_ratio": [0.75],
    # }
    #
    # create_comprehensive_table(file_path="./data_out/data_out_server_3.csv",
    #                            output_file="./data_out/analysis/table_test_1",
    #                            target="epsilon",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)

    filters_2 = {
        "num_students": [100],
        "num_schools": [2, 5, 8, 11, 14],
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    filters_2 = {
        "num_students": [100],
        "num_schools": [5, 8, 11, 14],
        "epsilon": [0.002],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.75],
    }

    create_comprehensive_table(file_path="./data_out/data_out_100_1.csv",
                               output_file="./data_out/analysis/table_test_1",
                               target="manipulators_ratio",
                               num_best=2,
                               only_gs=True,
                               filters=filters_2,
                               groupby="num_schools",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)

    filters_2 = {
        "num_students": [100],
        "num_schools": [5, 8, 11, 14],
        "epsilon": [0.005],
        "manipulators_ratio": [0.75],
        "num_manipulations_ratio": [0.5, 0.75, 1],
    }

    create_comprehensive_table(file_path="./data_out/data_out_100_1.csv",
                               output_file="./data_out/analysis/table_test_1",
                               target="num_manipulations_ratio",
                               num_best=2,
                               only_gs=True,
                               filters=filters_2,
                               groupby="num_schools",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)

    filters_2 = {
        "num_students": [100],
        "num_schools": [5, 8, 11, 14],
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.75],
        "num_manipulations_ratio": [0.75],
    }

    create_comprehensive_table(file_path="./data_out/data_out_100_1.csv",
                               output_file="./data_out/analysis/table_test_1",
                               target="epsilon",
                               num_best=2,
                               only_gs=True,
                               filters=filters_2,
                               groupby="num_schools",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)



    # Таблица ученики - школы - средняя лучшая длина списка
    filters = {
        "num_students": [500],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }
    filters = {
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }
    filters_2 = {
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }
    filters = {
        "num_students": [500],
        "num_schools": [5],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }



    filters_2 = {
        "num_students": [100],
        "num_schools": [2, 5, 8, 11, 14],
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    # utility_table_students_schools_k(file_path="./data_out/data_out_100_1.csv",
    #                                  output_file="./data_out/analysis/table_100_students_schools_k",
    #                                  num_best=1,
    #                                  only_gs=True,
    #                                  filters=None)

    # create_comprehensive_table(file_path="./data_out/data_out_100_1.csv",
    #                             output_file = "./data_out/analysis/table_test_1",
    #                             target = "num_manipulations_ratio",
    #                             num_best = 1,
    #                             only_gs = True,
    #                             filters = filters_2,
    #                             groupby = "num_schools",
    #                             show_k_ratio = True,
    #                             show_utility = True, show_unassigned = True,
    #                             show_fair = True, show_manipulator = True)








    # filters = {
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }

    # create_unassigned_students_table(file_path="./data_out/data_out_server_3.csv",
    #                                  output_file="./data_out/analysis/table_unassigned_students",
    #                                  num_best=1,
    #                                  only_gs=True,
    #                                  target="all",
    #                                  filters=filters)

