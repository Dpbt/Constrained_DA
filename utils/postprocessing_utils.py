import itertools
import json
import os

import pandas as pd


def find_missing_files(folder_path: str,
                       start: int,
                       end: int,
                       name_startswith: str = "test_results_",
                       name_endswith: str = ".csv"
                       ) -> None:
    """
    Finds and prints missing file numbers in a folder for files named 'test_results_<number>.csv'.

    This function scans the specified directory for files matching the pattern 'test_results_<number>.csv',
    extracts the numbers, and determines which numbers in the range [start, end] are missing.

    Parameters:
        folder_path (str): Path to the folder to scan.
        start (int): Start of the range (inclusive).
        end (int): End of the range (inclusive).
        name_startswith (str): Prefix of the file names to look for.
        name_endswith (str): Suffix of the file names to look for.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified folder does not exist.
    """
    files = os.listdir(folder_path)

    existing_numbers = set()
    for file in files:
        if file.startswith(name_startswith) and file.endswith(name_endswith):
            try:
                number = int(file.split("_")[-1].split(".")[0])
                existing_numbers.add(number)
            except ValueError:
                continue

    missing_numbers = [
        num for num in range(start, end + 1) if num not in existing_numbers
    ]

    print("Missing file numbers:", missing_numbers, len(missing_numbers))


def find_missing_experiments(input_file_path: str, params_lists: dict) -> str:
    """
    Identifies and reports missing experiment parameter combinations in a results CSV file.

    This function reads a CSV file containing experiment results, generates all possible parameter
    combinations from the provided params_lists, and checks which combinations are absent from the CSV.

    Parameters:
        input_file_path (str): Path to the CSV file with experiment results.
        params_lists (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        str: Formatted string listing all missing experiment configurations in JSON format,
             one per line, and the total count.

    Raises:
        FileNotFoundError: If the specified input_file_path does not exist.
        KeyError: If required keys are missing in params_lists.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    # Generate all possible parameter combinations
    keys = list(params_lists.keys())
    values = list(params_lists.values())
    all_combinations = list(itertools.product(*values))

    missing_experiments = []

    for combo in all_combinations:
        combo_dict = dict(zip(keys, combo))

        # Build query string for DataFrame filtering
        query = ' & '.join([f"({k} == {v})" for k, v in combo_dict.items()])
        if df.query(query).empty:
            missing_experiments.append(combo_dict)

    # Output formatting
    formatted_output = "Missing experiments:\n"
    for experiment in missing_experiments:
        formatted_output += json.dumps(experiment, separators=(',', ':')) + "\n"

    return formatted_output


def concatenate_csv_files(folder_path: str,
                          output_file_path: str,
                          name_startswith: str = "test_results_",
                          name_endswith: str = ".csv"
                          ) -> None:
    """
    Concatenates all CSV files in a folder matching the pattern 'test_results_*.csv' into a single CSV file.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        output_file_path (str): Path where the combined CSV will be saved.
        name_startswith (str): Prefix of the file names to look for.
        name_endswith (str): Suffix of the file names to look for.

    Returns:
        None
    """
    dataframes = []

    files = os.listdir(folder_path)

    for file in files:
        if file.startswith(name_startswith) and file.endswith(name_endswith):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined dataframe saved to {output_file_path}")


def filter_dataframe_by_conditions(input_file_path: str, output_file_path: str, filters: dict) -> None:
    """
    Loads a DataFrame from a CSV file and filters it according to the conditions specified in the dictionary.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Path where the filtered CSV will be saved.
        filters (dict): Dictionary where keys are column names and values are lists of values to filter by.

    Returns:
        None

    Features:
        1. Sequentially applies each filter in the provided dictionary.
        2. Prints the number of rows after each filter is applied.
        3. Handles missing columns gracefully with an informative message.
        4. Saves the filtered DataFrame to the specified output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the filters argument is not a dictionary.
    """
    df = pd.read_csv(input_file_path)

    for column, values in filters.items():
        if column in df.columns:
            current_count = len(df)
            df = df[df[column].isin(values)]
            print(f"Number of rows after filtering by '{column}': {len(df)} (was {current_count})")
        else:
            print(f"Column '{column}' not found in DataFrame.")

    df.to_csv(output_file_path, index=False)


def add_utility_rating_and_k_to_schools_ratio(input_file_path: str, output_file_path: str) -> None:
    """
    Adds utility ranking within each experiment and the k-to-schools ratio to a results DataFrame.

    This function reads a DataFrame from a CSV file, calculates a ranking of 'average_utility'
    within each 'experiment_number' group, and adds a new column 'utility_rating_in_experiment'.
    It also computes the ratio of 'k' to 'num_schools' for each row, rounded to three decimals,
    and saves the updated DataFrame to a new CSV file.

    Parameters:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to save the updated CSV file.

    Returns:
        None

    Features:
        1. Ranks 'average_utility' within each experiment (experiment_number).
        2. Adds a 'k_to_schools_ratio' column as k / num_schools, rounded to 3 decimals.
        3. Saves the updated DataFrame to the specified output file.
        4. Prints a confirmation message upon completion.

    Raises:
        FileNotFoundError: If the input file does not exist.
        KeyError: If required columns are missing in the input file.
    """
    # Read DataFrame from file
    df = pd.read_csv(input_file_path)

    # Add utility ranking within each experiment
    df['utility_rating_in_experiment'] = df.groupby('experiment_number')['average_utility'].rank(
        ascending=False, method='dense'
    )

    # Add k-to-schools ratio
    df['k_to_schools_ratio'] = (df['k'] / df['num_schools']).round(3)

    # Save the updated DataFrame to a new file
    df.to_csv(output_file_path, index=False)

    print(f"Updated DataFrame saved to file: {output_file_path}")


def usage_example():
    """
    Example usage of the functions in this module.

    This function demonstrates how to use the functions defined in this module.
    It includes examples for finding missing files, filtering DataFrames,
    and adding utility ratings and k-to-schools ratios.

    Returns:
        None
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    concatenate_csv_files(folder_path="../data_out/technical", output_file_path="../data_out/data_out_concat.csv")

    add_utility_rating_and_k_to_schools_ratio(input_file_path="../data_out/data_out_concat.csv",
                                              output_file_path="../data_out/data_out_concat.csv")

    find_missing_files(folder_path="../data_out/technical", start=0, end=25)

    # Missing experiments for 500
    params_lists = {
        "num_students": [100],
        "num_schools": [5, 10],
        "epsilon": [0.002, 0.01],
        "manipulators_ratio": [1],
        "num_manipulations_ratio": [0.75, 1],
    }
    missing = find_missing_experiments(input_file_path="../data_out/data_out_concat.csv",
                                       params_lists=params_lists)
    print("Number of missing experiments", len(missing))
    print(missing)

    # Filtering by parameters
    filters = {
        "num_students": [100, 200, 300, 400],
        "num_schools": [5, 10, 20],
        "epsilon": [0.002, 0.005, 0.01],
        "manipulators_ratio": [0.5, 0.75, 1],
        "num_manipulations_ratio": [0.5, 0.75, 1],
    }
    filter_dataframe_by_conditions(
        input_file_path="../data_out/data_out_concat.csv",
        output_file_path="../data_out/data_out_concat_filtered.csv",
        filters=filters,
    )


if __name__ == "__main__":
    usage_example()
