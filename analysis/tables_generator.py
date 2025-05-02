from typing import Optional, Any

import pandas as pd


def utility_table_students_schools_k(file_path: str,
                                     output_file: str,
                                     num_best: int = 1,
                                     only_gs: bool = False,
                                     filters: Optional[dict[str, list[Any]]] = None) -> None:
    """
    Generates and saves a pivot table showing the average k_to_schools_ratio for the best results, grouped by num_students and num_schools.

    This function reads data from a CSV file, filters it based on specified criteria (algorithm, custom filters),
    selects the top 'num_best' results for each experiment based on 'utility_rating_in_experiment',
    and creates a pivot table that displays the average 'k_to_schools_ratio' grouped by 'num_students' and 'num_schools'.
    The pivot table is then saved to a CSV file and printed to the console.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_file (str): Base path (without suffix) for saving the pivot table to a CSV file. The suffix will be automatically generated
                       based on the `num_best` and `only_gs` parameters.
        num_best (int, optional): Number of top results to consider per experiment. Default is 1.
        only_gs (bool, optional): If True, only considers results from the 'gs' algorithm. Default is False.
        filters (dict, optional): Dictionary of filter conditions. Keys are column names, and values are lists of allowed values.
                               For 'num_manipulations', the filter values are interpreted as ratios relative to 'num_schools'
                               (i.e., the actual filtering is done against `round(ratio * num_schools)`).

    Returns:
        None

    Features:
        1. Flexible filtering based on algorithm and other experiment parameters.
        2. Selection of top N results per experiment based on utility ranking.
        3. Generation of a clear pivot table summarizing key relationships.
        4. Automatic filling of missing values with a dash ('-').
        5. Saves the pivot table as a CSV file.
        6. Customizable output file naming based on analysis parameters.

    Example:
        > utility_table_students_schools_k(
        ...     "results.csv", "output_table", num_best=3, only_gs=True,
        ...     filters={"num_students": [100, 200], "epsilon": [0.01, 0.1]}
        ... )
        num_schools           5      10
        num_students
        100                  0.45    0.55
        200                  0.6     0.7
        Table successfully saved to output_table_3_gs

    Implementation Notes:
        - Requires columns: 'experiment_number', 'utility_rating_in_experiment', 'k_to_schools_ratio',
          'num_students', 'num_schools', 'algorithm' to exist in the CSV file.
        - The output file name is constructed as `output_file + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"`.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        KeyError: If any required columns are missing from the DataFrame.
    """
    # Read data from the file
    df = pd.read_csv(file_path)

    if only_gs:
        # Filter data for GS algorithm only
        df = df[df['algorithm'] == 'gs']

    # Apply additional filters
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                if column == 'num_manipulations':
                    # Special handling for num_manipulations
                    mask = pd.Series(False, index=df.index)
                    for value in values:
                        mask |= (df['num_manipulations'] == round(value * df['num_schools']))
                    df = df[mask]
                else:
                    # Standard filtering
                    df = df[df[column].isin(values)]

    # Sort data by experiment and utility rating
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Select the top num_best results for each experiment
    best_utility_df = df.groupby('experiment_number').head(num_best)

    # Group data by number of students and schools, calculate average k_to_schools_ratio
    pivot_table = best_utility_df.groupby(['num_students', 'num_schools'])['k_to_schools_ratio'].mean().unstack()

    # Sort indices and columns
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Fill NaN values with dashes
    pivot_table = pivot_table.fillna('-')

    # Create output file name
    output_file = output_file + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"
    # Save the table to the specified file
    pivot_table.to_csv(output_file, index=True)

    print(pivot_table)
    print(f"Table successfully saved to {output_file}")


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
                unassigned_parts.append(
                    f"F: {row['average_percentage_unassigned_fair_students'].iloc[0]:.{accuracy}f}%")
            if show_manipulator:
                unassigned_parts.append(
                    f"M: {row['average_percentage_unassigned_manipulator_students'].iloc[0]:.{accuracy}f}%")
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


def create_comprehensive_table_k_lists(file_path, output_file, target, only_gs=False, filters=None,
                                       groupby="num_students"):
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

    # Сортировка данных по utility_rating_in_experiment (по убыванию)
    df = df.sort_values('utility_rating_in_experiment', ascending=True)

    # Группировка данных по target и groupby
    groupby_columns = [target]
    if groupby:
        groupby_columns.append(groupby)

    # Создание списка значений k в порядке убывания utility_rating_in_experiment
    def collect_k_values(group):
        return tuple(group['k'].tolist())  # Собираем значения k в кортеж

    result_df = df.groupby(groupby_columns).apply(collect_k_values).reset_index()
    result_df.columns = groupby_columns + ['k_list']  # Переименовываем столбцы

    # Сохранение результата в файл
    output_file = f"{output_file}_{'gs' if only_gs else 'all'}_{target}.csv"
    result_df.to_csv(output_file, index=False)

    print(result_df)
    print(f"Таблица успешно сохранена в {output_file}")


def get_n_best_results(file_path: str, n: int = 1) -> tuple[pd.DataFrame, float]:
    """
    Retrieves the top n results per experiment based on average utility from a CSV file.

    This function reads experiment results from the specified CSV file, groups the data by
    'experiment_number', and selects the top n rows with the highest 'average_utility' within
    each group. It also calculates the mean of the 'k_to_schools_ratio' column for these top results.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.
        n (int, optional): Number of top results to select per experiment. Default is 1.

    Returns:
        Tuple[pd.DataFrame, float]:
            - DataFrame containing the top n results per experiment with selected columns:
              ['experiment_number', 'num_students', 'num_schools', 'epsilon', 'algorithm', 'k',
               'average_utility', 'k_to_schools_ratio']
            - Mean value of 'k_to_schools_ratio' across the selected top results.

    Notes:
        - Assumes the CSV file contains all the columns listed in the result.
        - Uses pandas groupby and nlargest for efficient selection.
    """
    df = pd.read_csv(file_path)
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

    best_k_to_schools_ratio_mean = result["k_to_schools_ratio"].mean()

    return result, best_k_to_schools_ratio_mean


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Лучшие результаты и оптимальная длина списка
    # input_file_path = "../data_out/data_out_100.csv"
    # # input_file_path = "../data_out/data_out_server.csv"
    # top_results, best_k_to_schools_ratio_mean = get_n_best_results(input_file_path=input_file_path, n=1)
    # print(top_results)
    # print(best_k_to_schools_ratio_mean)

    # Список колонок
    # df = pd.read_csv("../data_out/data_out_server_3_filtered.csv")
    # column_list = df.columns.tolist()
    # print(column_list)

    # 100 - 8, 500 - 5

    # Part 100-500

    filters = {
        "num_students": [100, 200, 300, 400, 500],
        "num_schools": [5, 10, 20],
        "epsilon": [0.005],
        "manipulators_ratio": [0.75],
        "num_manipulations": [0.75],
    }

    filters = {
        "num_students": [100, 200, 300, 400, 500],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    utility_table_students_schools_k(file_path="../data_out/data_out_server.csv",
                                     output_file="../data_out/analysis/table_filtered_students_schools_k",
                                     num_best=1,
                                     only_gs=True,
                                     filters=filters)

    filters_2 = {
        "num_students": [100],
        "num_schools": [2, 5, 8, 11, 14],
        "epsilon": [0.002],
        "manipulators_ratio": [0.75],
        "num_manipulations": [0.75],
    }

    filters_2 = {
        "num_students": [100],
        "num_schools": [2, 5, 8, 11, 14],
        "epsilon": [0.001, 0.002, 0.005, 0.01],
        "manipulators_ratio": [0.25, 0.5, 0.75, 1],
        "num_manipulations": [0.5, 0.75, 1],
    }

    utility_table_students_schools_k(file_path="../data_out/data_out_100.csv",
                                     output_file="../data_out/analysis/table_filtered_students_schools_k",
                                     num_best=1,
                                     only_gs=True,
                                     filters=filters_2)

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
    # create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="manipulators_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="manipulators_ratio",
    #                                    only_gs=True,
    #                                    filters=filters,
    #                                    groupby="num_students")
    #
    #
    # filters = {
    #     "num_students": [400],
    #     "num_schools": [5],
    #     "epsilon": [0.005],
    #     "manipulators_ratio": [0.75],
    #     "num_manipulations_ratio": [0.5, 0.75, 1],
    # }
    #
    # create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="num_manipulations_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="num_manipulations_ratio",
    #                                    only_gs=True,
    #                                    filters=filters,
    #                                    groupby="num_students")
    #
    # filters = {
    #     "num_students": [400],
    #     "num_schools": [5],
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [1],
    #     "num_manipulations_ratio": [1],
    # }
    #
    # create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="epsilon",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters,
    #                            groupby="num_students",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="epsilon",
    #                                    only_gs=True,
    #                                    filters=filters,
    #                                    groupby="num_students")
    #
    # filters_2 = {
    #     "num_students": [100],
    #     "num_schools": [2, 5, 8, 11, 14],
    #     "epsilon": [0.001, 0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.25, 0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }

    # filters_2 = {
    #     "num_students": [100],
    #     "num_schools": [5],
    #     "epsilon": [0.002],
    #     "manipulators_ratio": [0.25, 0.5, 0.75, 1],
    #     "num_manipulations": [0.75],
    # }
    #
    # create_comprehensive_table(input_file_path="../data_out/data_out_100.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="manipulators_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters_2,
    #                            groupby="num_schools",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_100.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="manipulators_ratio",
    #                                    only_gs=True,
    #                                    filters=filters_2,
    #                                    groupby="num_students")
    #
    # filters_2 = {
    #     "num_students": [100],
    #     "num_schools": [5],
    #     "epsilon": [0.002],
    #     "manipulators_ratio": [0.75],
    #     "num_manipulations_ratio": [0.5, 0.75, 1],
    # }
    #
    # create_comprehensive_table(input_file_path="../data_out/data_out_100.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="num_manipulations_ratio",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters_2,
    #                            groupby="num_schools",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_100.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="num_manipulations_ratio",
    #                                    only_gs=True,
    #                                    filters=filters_2,
    #                                    groupby="num_students")
    #
    # filters_2 = {
    #     "num_students": [100],
    #     "num_schools": [5],
    #     "epsilon": [0.001, 0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.75],
    #     "num_manipulations_ratio": [0.75],
    # }
    #
    # create_comprehensive_table(input_file_path="../data_out/data_out_100.csv",
    #                            output_file_path="../data_out/analysis/table_test_1",
    #                            target="epsilon",
    #                            num_best=2,
    #                            only_gs=True,
    #                            filters=filters_2,
    #                            groupby="num_schools",
    #                            show_k_ratio=True,
    #                            show_utility=True, show_unassigned=True,
    #                            show_fair=True, show_manipulator=True)
    #
    # create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_100.csv",
    #                                    output_file_path="../data_out/analysis/table_test_1",
    #                                    target="epsilon",
    #                                    only_gs=True,
    #                                    filters=filters_2,
    #                                    groupby="num_students")

    # filters = {
    #     "epsilon": [0.002, 0.005, 0.01],
    #     "manipulators_ratio": [0.5, 0.75, 1],
    #     "num_manipulations": [0.5, 0.75, 1],
    # }

    # create_unassigned_students_table(input_file_path="../data_out/data_out_server.csv",
    #                                  output_file_path="../data_out/analysis/table_unassigned_students",
    #                                  num_best=1,
    #                                  only_gs=True,
    #                                  target="all",
    #                                  filters=filters)
