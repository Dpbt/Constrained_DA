import pandas as pd
import numpy as np
import os
import itertools
import json


def find_missing_files(folder_path, start, end):
    files = os.listdir(folder_path)

    existing_numbers = set()
    for file in files:
        if file.startswith("test_results_") and file.endswith(".csv"):
            try:
                number = int(file.split("_")[-1].split(".")[0])
                existing_numbers.add(number)
            except ValueError:
                continue

    missing_numbers = [
        num for num in range(start, end + 1) if num not in existing_numbers
    ]

    print("Отсутствующие номера файлов:", missing_numbers, len(missing_numbers))


def find_missing_experiments(file_path, tests_lists):
    # Чтение CSV файла
    df = pd.read_csv(file_path)

    # Создание всех возможных комбинаций параметров
    keys = list(tests_lists.keys())
    values = list(tests_lists.values())
    all_combinations = list(itertools.product(*values))

    missing_experiments = []

    for combo in all_combinations:
        combo_dict = dict(zip(keys, combo))
        num_schools = combo_dict['num_schools']
        num_manipulations = combo_dict['num_manipulations']

        # Преобразование num_manipulations
        combo_dict['num_manipulations'] = round(num_manipulations * num_schools)

        # Проверка наличия комбинации в DataFrame
        query = ' & '.join([f"({k} == {v})" for k, v in combo_dict.items()])
        if df.query(query).empty:
            missing_experiments.append(combo_dict)

    print(len(missing_experiments))
    # Форматирование вывода
    formatted_output = "Отсутствующие эксперименты:\n"
    for experiment in missing_experiments:
        formatted_output += json.dumps(experiment, separators=(',', ':')) + "\n"

    return formatted_output


def concatenate_csv_files(folder_path, output_file):
    dataframes = []

    files = os.listdir(folder_path)

    for file in files:
        if file.startswith("test_results_") and file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    # print(combined_df.columns)
    print(f"Объединенные данные сохранены в {output_file}")


def filter_dataframe_by_conditions(file_path, output_file, filters):
    """
    Загружает DataFrame из CSV файла и фильтрует его по условиям из словаря.

    :param file_path: Путь к CSV файлу
    :param filters: Словарь, где ключи - названия столбцов, а значения - списки значений для фильтрации
    :return: Отфильтрованный DataFrame
    """

    df = pd.read_csv(file_path)

    for column, values in filters.items():
        if column in df.columns:
            current_count = len(df)

            if column == 'num_manipulations':
                mask = pd.Series(False, index=df.index)
                for value in values:
                    mask |= (df['num_manipulations'] == round(value * df['num_schools']))
                df = df[mask]
            else:
                df = df[df[column].isin(values)]

            print(f"Количество строк после фильтрации по '{column}': {len(df)} (было {current_count})")
        else:
            print(f"Столбец '{column}' не найден в DataFrame.")

    df.to_csv(output_file, index=False)


def get_n_best_results(file_path: str, n=1):
    df = pd.read_csv(file_path)
    df["k_to_schools_ratio"] = df["k"] / df["num_schools"]
    grouped = df.groupby("experiment_number").apply(
        lambda x: x.nlargest(n, "average_utility")
    )
    grouped = grouped.reset_index(drop=True)
    result = grouped[
        [
            "experiment_number",
            "num_students",
            "num_schools",
            "epsilon",
            "algorithm",
            "k",
            "average_utility",
            "k_to_schools_ratio",
        ]
    ]
    result["k_to_schools_ratio"] = result["k_to_schools_ratio"].round(3)

    best_k_to_schools_ratio_mean = result["k_to_schools_ratio"].mean()

    return result, best_k_to_schools_ratio_mean


def add_columns_and_save(input_file, output_file):
    # Чтение DataFrame из файла
    df = pd.read_csv(input_file)

    # Добавление столбца k_to_schools_ratio
    df['k_to_schools_ratio'] = df['k'] / df['num_schools']

    # Добавление столбца utility_rating_in_experiment
    df['utility_rating_in_experiment'] = df.groupby('experiment_number')['average_utility'].rank(ascending=False,
                                                                                                 method='dense')

    # Сохранение обновленного DataFrame в новый файл
    df.to_csv(output_file, index=False)

    print(f"Обновленный DataFrame сохранен в файл: {output_file}")

    return df


def add_num_manipulations_ratio(input_file_path, output_file_path):
    # Чтение данных из входного файла
    df = pd.read_csv(input_file_path)

    # Функция для определения num_manipulations_ratio
    def determine_ratio(num_manipulations, num_schools):
        if num_schools == 0:
            return np.nan
        actual_ratio = num_manipulations / num_schools
        possible_ratios = [0.25, 0.5, 0.75, 1.0]
        return min(possible_ratios, key=lambda x: abs(x - actual_ratio))

    # Добавляем новый столбец num_manipulations_ratio
    df['num_manipulations_ratio'] = df.apply(lambda row: determine_ratio(row['num_manipulations'], row['num_schools']), axis=1)

    # Сохранение обновленного DataFrame в выходной файл
    df.to_csv(output_file_path, index=False)

    print(f"Обработка завершена. Результаты сохранены в {output_file_path}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    tests_lists = {
        "num_students": [500],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }
    tests_lists = {
        "num_students": [100],
        "num_schools": [2, 5, 8, 11, 14],
        "num_capacities": [5],
        "num_repeats_profiles": [5],
        "num_repeat_sampler": [50],
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }
    test_list_100 = {
        "num_students": [100],
        # "num_schools": [2, 5, 8, 11, 14],
        # "num_capacities": [5],
        # "num_repeats_profiles": [5],
        # "num_repeat_sampler": [50],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    # concatenate_csv_files(folder_path="./data_out/technical", output_file="data_out/data_out_100_1.csv")

    # find_missing_files(folder_path="./data_out_server", start=0, end=400)
    # find_missing_files(folder_path="./data_out/technical", start=0, end=239)

    # Отсутствующие эксперименты для 500
    # file_path = "data_out/data_out_server.csv"
    # tests_lists = {
    #     "num_students": [500],
    #     "num_schools": [5, 10, 20],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }
    # missing = find_missing_experiments(file_path, tests_lists)
    # print(missing)

    # Лучшие результаты и оптимальная длина списка
    # file_path = "data_out/data_out_100_1.csv"
    # file_path = "data_out/data_out_server.csv"
    # file_path = "./data_out/data_out_100_filtered.csv"
    # top_results, best_k_to_schools_ratio_mean = get_n_best_results(file_path=file_path, n=1)
    # print(top_results)
    # print(best_k_to_schools_ratio_mean)

    # Фильтрация по параметрам
    # filters = {
    #     "num_students": [100, 200, 300, 400],
    #     "num_schools": [5, 10, 20],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }
    # filter_dataframe_by_conditions(
    #     file_path="./data_out/data_out_server.csv",
    #     output_file="./data_out/data_out_server_3_filtered.csv",
    #     filters=filters,
    # )

    # add_columns_and_save(input_file="./data_out/data_out_100_1.csv",
    #                      output_file="./data_out/data_out_100_1.csv")

    # add_num_manipulations_ratio(input_file_path="./data_out/data_out_100_1.csv",
    #                             output_file_path="./data_out/data_out_100_1.csv")




