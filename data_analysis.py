import pandas as pd
import os


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

    print("Отсутствующие номера файлов:", missing_numbers)


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


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # find_missing_files(folder_path="./data_out_server_2", start=0, end=404)

    # concatenate_csv_files(folder_path="./data_out_server_2", output_file="./data_out/data_out_server_2.csv")

    # tests_lists = {
    #     "num_students": [100, 200, 300, 400, 500],
    #     "num_schools": [5, 10, 20],
    #     "num_capacities": [5],
    #     "num_repeats_profiles": [5],
    #     "num_repeat_sampler": [50],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }

    filters = {
        "num_students": [100, 200, 300, 400],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    filter_dataframe_by_conditions(
        file_path="./data_out/data_out_server_2.csv",
        output_file="./data_out/data_out_server_2_filtered.csv",
        filters=filters,
    )

    file_path = './data_out/data_out_server_2_filtered.csv'
    top_results, best_k_to_schools_ratio_mean = get_n_best_results(file_path=file_path, n=1)
    print(top_results)
    print(best_k_to_schools_ratio_mean)



