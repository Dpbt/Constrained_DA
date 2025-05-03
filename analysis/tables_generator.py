from typing import Optional, Any

import pandas as pd


def create_utility_table_students_schools_k(input_file_path: str,
                                            output_file_path: str,
                                            num_best: int = 1,
                                            only_gs: bool = False,
                                            filters: Optional[dict[str, list[Any]]] = None) -> None:
    """
    Generates and saves a pivot table showing the average k_to_schools_ratio for the best results, grouped by num_students and num_schools.

    This function reads data from a CSV file, filters it based on specified criteria,
    selects the top 'num_best' results for each experiment based on 'utility_rating_in_experiment',
    and creates a pivot table that displays the average 'k_to_schools_ratio' grouped by 'num_students' and 'num_schools'.
    The pivot table is then saved to a CSV file and printed to the console.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Base path (without suffix) for saving the pivot table to a CSV file. The suffix will be automatically generated
                       based on the `num_best` and `only_gs` parameters.
        num_best (int, optional): Number of top results (by utility) to select per experiment. (Default: 1).
        only_gs (bool, optional): If True, only considers results from the 'gs' algorithm. (Default: False).
        filters (dict, optional): Dictionary of filter conditions. Keys are column names, and values are lists of allowed values. (Default: None)

    Returns:
        None

    Implementation Notes:
        - Requires columns: 'experiment_number', 'utility_rating_in_experiment', 'k_to_schools_ratio',
          'num_students', 'num_schools', 'algorithm' to exist in the CSV file.
        - The output file name is constructed as `output_file_path + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"`.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        KeyError: If any required columns are missing from the DataFrame.
    """
    # Read data from the file
    df = pd.read_csv(input_file_path)

    if only_gs:
        # Filter data for GS algorithm only
        df = df[df['algorithm'] == 'gs']

    # Apply additional filters
    if filters:
        for column, values in filters.items():
            if column in df.columns:
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
    output_file_path = output_file_path + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"
    # Save the table to the specified file
    pivot_table.to_csv(output_file_path, index=True)

    print(pivot_table)
    print(f"Table successfully saved to {output_file_path}")


def create_unassigned_students_table(
        input_file_path: str,
        output_file_path: str,
        num_best: int = 1,
        only_gs: bool = False,
        target: str = "all",
        filters: Optional[dict[str, list[Any]]] = None
) -> None:
    """
    Generates and saves a pivot table of average unassigned student percentages for the best results,
    grouped by num_students and num_schools.

    This function reads experiment results from a CSV file, filters the data according to the specified
    algorithm, target group (all, fair, or manipulators), and any additional filters. It then selects
    the top 'num_best' results for each experiment based on utility ranking and creates a pivot table
    showing the mean percentage of unassigned students for the chosen group. The pivot table is saved
    as a CSV file and printed to the console.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Base path (without suffix) for saving the pivot table to a CSV file. The suffix will be automatically generated
                       based on the `num_best` and `only_gs` parameters.
        num_best (int, optional): Number of top results (by utility) to select per experiment. (Default: 1).
        only_gs (bool, optional): If True, only considers results from the 'gs' algorithm. (Default: False).
        target (str, optional): Which group of students to analyze unassigned percentage for.
                                Must be one of {"all", "fair", "manipulators"}. (Default: "all").
        filters (dict, optional): Dictionary of filter conditions. Keys are column names, and values are lists of allowed values. (Default: None)

    Returns:
        None

    Implementation Notes:
        - Requires columns: 'experiment_number', 'utility_rating_in_experiment', target column,
          'num_students', 'num_schools', 'algorithm' to exist in the CSV file.
        - The output file name is constructed as `output_file_path + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"`.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        KeyError: If required columns are missing from the DataFrame.
        ValueError: If `target` is not one of {"all", "fair", "manipulators"}.
    """
    # Read data from the file
    df = pd.read_csv(input_file_path)

    target_dict = {
        "all": "average_percentage_unassigned_students",
        "fair": "average_percentage_unassigned_fair_students",
        "manipulators": "average_percentage_unassigned_manipulator_students",
    }

    if target not in target_dict:
        raise ValueError("target must be one of {'all', 'fair', 'manipulators'}")

    target_column = target_dict[target]

    if only_gs:
        # Filter data for GS algorithm only
        df = df[df['algorithm'] == 'gs']

    # Apply additional filters
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]

    # Sort data by experiment and utility rating
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Select the top num_best results for each experiment
    best_utility_df = df.groupby('experiment_number').head(num_best)

    # Group data by number of students and schools, calculate average unassigned percentage
    pivot_table = best_utility_df.groupby(['num_students', 'num_schools'])[target_column].mean().unstack()

    # Sort indices and columns
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Fill NaN values with dashes
    pivot_table = pivot_table.fillna('-')

    # Create output file name
    output_file_path = output_file_path + f"_{num_best}" + f"_{'gs' if only_gs else 'all'}"
    # Save the table to the specified file
    pivot_table.to_csv(output_file_path, index=True)

    print(pivot_table)
    print(f"Table successfully saved to {output_file_path}")


def create_comprehensive_table(
        input_file_path: str,
        output_file_path: str,
        target: str,
        num_best: int = 1,
        only_gs: bool = False,
        filters: Optional[dict[str, list[Any]]] = None,
        groupby: str = "num_students",
        show_k_ratio: bool = True,
        show_utility: bool = True,
        show_unassigned: bool = True,
        show_fair: bool = True,
        show_manipulator: bool = True
) -> None:
    """
    Generates and saves a comprehensive summary table of experiment results with flexible aggregation and formatting.

    This function reads experiment results from a CSV file, applies filtering by algorithm and custom parameters,
    selects the top N results per experiment by utility ranking, groups and aggregates the data by the specified axes,
    and formats each cell to display multiple metrics (such as k-to-schools ratio, utility, and unassigned percentages).
    The resulting table is saved as a CSV file and printed to the console.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Base name for the output CSV file; suffixes are added automatically (details in the implementation notes).
        target (str): Name of the column to use as the primary grouping axis (e.g., 'num_schools').
        num_best (int, optional): Number of top results (by utility) to select per experiment. (Default: 1).
        only_gs (bool, optional): If True, only considers results from the 'gs' algorithm. (Default: False).
        filters (dict, optional): Dictionary of filter conditions. Keys are column names, and values are lists of allowed values. (Default: None)
        groupby (str, optional): Name of the column for the secondary grouping axis. (Default: 'num_students')
        show_k_ratio (bool, optional): Whether to display the k-to-schools ratio in the table. (Default: True)
        show_utility (bool, optional): Whether to display utility metrics in the table. (Default: True)
        show_unassigned (bool, optional): Whether to display unassigned student percentages in the table. (Default: True)
        show_fair (bool, optional): Whether to include metrics for fair students. (Default: True)
        show_manipulator (bool, optional): Whether to include metrics for manipulators. (Default: True)

    Returns:
        None

    Implementation Notes:
        - Requires columns: 'experiment_number', 'utility_rating_in_experiment', 'k_to_schools_ratio',
          'num_students', 'num_schools', 'algorithm', and others depending on selected metrics.
        - The output file name is formatted as:
          output_file_path + f"_{num_best}_{'gs' if only_gs else 'all'}_{target}.csv"
        - Aggregation is performed using groupby on [target, groupby] and mean on selected metrics.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        KeyError: If required columns are missing from the DataFrame.
        ValueError: If the target column is not present in the DataFrame.
    """
    # Load data from file
    df = pd.read_csv(input_file_path)

    if only_gs:
        # Filter data for GS algorithm only
        df = df[df['algorithm'] == 'gs']

    # Apply additional filters
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]

    # Sort data by experiment and utility ranking
    df = df.sort_values(['experiment_number', 'utility_rating_in_experiment'])

    # Select the top num_best results for each experiment
    best_utility_df = df.groupby('experiment_number').head(num_best)

    # Determine columns for aggregation
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

    # Group and aggregate data
    groupby_columns = []
    if groupby:
        groupby_columns.append(groupby)
    grouped = best_utility_df.groupby([target] + groupby_columns)
    agg_data = grouped[agg_columns].mean()

    # Format data for output
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

    # Sort indices and columns
    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Fill NaN values with dashes
    pivot_table = pivot_table.fillna('-')

    output_file_path = f"{output_file_path}_{num_best}_{'gs' if only_gs else 'all'}_{target}.csv"
    # Save the table to the specified file
    pivot_table.to_csv(output_file_path, index=True)

    print(pivot_table)
    print(f"Table successfully saved to {output_file_path}")


def create_comprehensive_table_k_lists(
        input_file_path: str,
        output_file_path: str,
        target: str,
        only_gs: bool = False,
        filters: Optional[dict[str, list[Any]]] = None,
        groupby: str = "num_students"
) -> None:
    """
    Generates and saves a table with lists of k values for each group, ordered by utility ranking.
    (i.e., outputs the lengths of the preference lists in descending order of average utility for them)

    This function reads experiment results from a CSV file, filters the data by algorithm and custom parameters,
    sorts by utility rating, groups by the specified axes, and collects the sequence of k values for each group.
    The resulting table is saved as a CSV file and printed to the console.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Base name for the output CSV file; suffixes are added automatically (details in the implementation notes).
        target (str): Name of the column to use as the primary grouping axis (e.g., 'num_schools').
        only_gs (bool, optional): If True, only considers results from the 'gs' algorithm. (Default: False).
        filters (dict, optional): (dict, optional): Dictionary of filter conditions. Keys are column names, and values are lists of allowed values. (Default: None)
        groupby (str, optional): Name of the column for the secondary grouping axis. (Default: 'num_students')

    Returns:
        None

    Implementation Notes:
        - Requires columns: 'algorithm', 'utility_rating_in_experiment', 'k', and the specified grouping columns.
        - The output file name is formatted as:
          output_file_path + f"_{'gs' if only_gs else 'all'}_{target}.csv"
        - For 'num_manipulations', filter values are multiplied by 'num_schools' and rounded to integer.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        KeyError: If required columns are missing from the DataFrame.
        ValueError: If the target column is not present in the DataFrame.
    """
    # Load data from file
    df = pd.read_csv(input_file_path)

    if only_gs:
        # Filter data for GS algorithm only
        df = df[df['algorithm'] == 'gs']

    # Apply additional filters
    if filters:
        for column, values in filters.items():
            if column in df.columns:
                df = df[df[column].isin(values)]

    # Sort data by utility_rating_in_experiment (ascending order: best first)
    df = df.sort_values('utility_rating_in_experiment', ascending=True)

    # Group data by target and groupby columns
    groupby_columns = [target]
    if groupby:
        groupby_columns.append(groupby)

    # Create a list of k values in the order of utility rating for each group
    def collect_k_values(group):
        return tuple(group['k'].tolist())

    result_df = df.groupby(groupby_columns).apply(collect_k_values).reset_index()
    result_df.columns = groupby_columns + ['k_list']

    # Save the result to a file
    output_file_path = f"{output_file_path}_{'gs' if only_gs else 'all'}_{target}.csv"
    result_df.to_csv(output_file_path, index=False)

    print(result_df)
    print(f"Table successfully saved to {output_file_path}")


def get_n_best_results(input_file_path: str, n: int = 1) -> tuple[pd.DataFrame, float]:
    """
    Retrieves the top n results per experiment based on average utility from a CSV file.

    This function reads experiment results from the specified CSV file, groups the data by
    'experiment_number', and selects the top n rows with the highest 'average_utility' within
    each group. It also calculates the mean of the 'k_to_schools_ratio' column for these top results.

    Parameters:
        input_file_path (str): Path to the input CSV file.
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
    df = pd.read_csv(input_file_path)
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


def generate_tables_examples():
    """
    Generates example tables for demonstration. The tables used in the paper are generated as examples.

    Returns:
        None
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Utility table for students and schools
    filters = {
        "num_students": [100, 200, 300, 400, 500],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations_ratio": [0.5, 0.75, 1],
    }

    create_utility_table_students_schools_k(input_file_path="../data_out/data_out_server.csv",
                                            output_file_path="../data_out/analysis/table_students_schools_k",
                                            num_best=1,
                                            only_gs=True,
                                            filters=filters)

    # Table for manipulators_ratio
    filters = {
        "num_students": [400],
        "num_schools": [5],
        "epsilon": [0.005],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations_ratio": [0.75],
    }

    create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
                               output_file_path="../data_out/analysis/table_400",
                               target="manipulators_ratio",
                               num_best=2,
                               only_gs=True,
                               filters=filters,
                               groupby="num_students",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)

    create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
                                       output_file_path="../data_out/analysis/table_400_lists",
                                       target="manipulators_ratio",
                                       only_gs=True,
                                       filters=filters,
                                       groupby="num_students")

    # Table for num_manipulations_ratio
    filters = {
        "num_students": [400],
        "num_schools": [5],
        "epsilon": [0.005],
        "manipulators_ratio": [0.75],
        "num_manipulations_ratio": [0.5, 0.75, 1],
    }

    create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
                               output_file_path="../data_out/analysis/table_400",
                               target="num_manipulations_ratio",
                               num_best=2,
                               only_gs=True,
                               filters=filters,
                               groupby="num_students",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)

    create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
                                       output_file_path="../data_out/analysis/table_400_lists",
                                       target="num_manipulations_ratio",
                                       only_gs=True,
                                       filters=filters,
                                       groupby="num_students")

    # Table for epsilon
    filters = {
        "num_students": [400],
        "num_schools": [5],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [1],
        "num_manipulations_ratio": [1],
    }

    create_comprehensive_table(input_file_path="../data_out/data_out_server.csv",
                               output_file_path="../data_out/analysis/table_400",
                               target="epsilon",
                               num_best=2,
                               only_gs=True,
                               filters=filters,
                               groupby="num_students",
                               show_k_ratio=True,
                               show_utility=True, show_unassigned=True,
                               show_fair=True, show_manipulator=True)

    create_comprehensive_table_k_lists(input_file_path="../data_out/data_out_server.csv",
                                       output_file_path="../data_out/analysis/table_400_lists",
                                       target="epsilon",
                                       only_gs=True,
                                       filters=filters,
                                       groupby="num_students")

    # Unassigned students table
    filters = {
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations_ratio": [0.5, 0.75, 1],
    }

    create_unassigned_students_table(input_file_path="../data_out/data_out_server.csv",
                                     output_file_path="../data_out/analysis/table_unassigned_students",
                                     num_best=1,
                                     only_gs=True,
                                     target="all",
                                     filters=filters)


if __name__ == "__main__":
    generate_tables_examples()
