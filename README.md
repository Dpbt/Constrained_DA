# Constrained_DA

## Project Overview

This project contains the simulation code used for a research article. 
The simulations focus on school choice mechanisms, exploring the effects 
of various parameters on student assignments and utilities. The project 
includes tools for running experiments, analyzing results, and generating 
data for further research.

---

## Project Structure 

- **`algorithms/`**: Contains the implementation of school choice mechanisms and manipulation algorithms
  - **`boston.py`**: Implementation of the Boston mechanism with list length k
  - **`chinese_parallel.py`**: Implementation of Chinese parallel mechanism with list length k at each round
  - **`gale_shapley.py`**: Implementation of Deferred Acceptance mechanism with list length k
  - **`manipulation.py`**: Algorithm for modeling student manipulation
  - **`probs_estimator.py`**: Function to estimate students' probability of getting into each school based on preference lists
  - **`sampler.py`**: Function for multiple simulations of the selected mechanism for preference lists submitted by students
- **`utils/`**: Contains auxiliary functions for the other modules
  - **`algorithm_enums.py`**: Enums for supported algorithms
  - **`experiment_utils.py`**: Functions for job generation, dataframe grouping and result generation in the test system
  - **`generation_utils.py`**: Functions for generating profiles, capacities, lists
  - **`postprocessing_utils.py`**: Functions that can be useful during and after simulations before analyzing the data
  - **`statistic_utils.py`**: Functions for generating statistics for preference lists and unassigned statistics for results
  - **`utilities.py`**: Functions for calculating the expected utility of students 
- **`analysis/`**: Contains functions for processing results, building tables for analysis, and a test system that checks all available preference lists and searches for a Nash equilibrium among them
  - **`all_preferences_test_system.py`**: Test system that tests all available preference lists and searches for a Nash equilibrium (optionally symmetric) among them. It is convenient for analyzing small examples
  - **`nash_equilibrium_searcher.py`**: Function for finding Nash equilibria (optionally symmetric)
  - **`tables_generator.py`**: Functions for generating some tables for analyzing simulation results
- **`data_out/`**: Directory for storing experiment results
- **`test_system.py`**: Core logic for running simulations
- **`run_experiment.py`**: Basic script for setting parameters and running simulations

---

## Requirements

* We use Python 3.12.10. Other dependencies are in the **`requirements.txt`** file. A lower or higher version of Python and libraries may also work.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dpbt/Constrained_DA
   cd <repository_directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
---

## Usage

### Running Experiments

The main entry point for running experiments is `run_experiment.py`. You can configure and execute experiments by providing parameter dictionaries.

#### Example Command:

```bash
python run_experiment.py
```

#### Example of parameter setting:

```
params_lists = {
    "num_students": [12, 14],
    "num_schools": [3, 5, 8],
    "num_capacities": [5],
    "num_repeats_profiles": [5],
    "num_repeat_sampler": [50],
    "epsilon": [0.001],
    "manipulators_ratio": [0.75, 1.0],
    "num_manipulations_ratio": [0.75, 1.0],
}
```

#### Parameters:
- **`num_students`**: (int) Total number of students.
- **`num_schools`**: (int) Total number of available schools.
- **`capacities`**: (np.array) School capacities (shape: (num_schools,)) (optional, generated if not provided).
- **`num_capacities`**: (int) Number of capacity variants to generate (optional, if capacities specified will be set as 1) (default: 1).
- **`num_repeats_profiles`**: (int) Number of random preference profiles to simulate (default: 10).
- **`num_repeat_sampler`**: (int) Number of repetitions for each mechanism  generated preference lists (default: 1000).
- **`epsilon`**: (float) Minimum utility improvement threshold (default: 0.01).
- **`manipulators_ratio`**: (float) Fraction of students allowed to manipulate (default: 1.0, i.e., all students will be able to manipulate).
- **`num_manipulations_ratio`**: (float) Fraction of allowed manipulations from the num_schools parameter (default: 1.0, i.e., students will be allowed to perform num_schools manipulations).

#### Output:
The results are saved as CSV files in the `data_out` directory. Each file contains detailed experiment results, including:
- Average utilities
- Manipulation statistics
- Unassigned student percentages
- Algorithm-specific metrics

---

## Example Workflow

1. **Set Parameters**: Define parameter dictionaries in `run_experiment.py` or directly in the script.
2. **Run Experiments**: Execute the script to generate results.
3. **Analyze Results**: Use `tables_generator.py` to build tables or extract best results.

---

### References

* [1] Balancing Incentives and Efficiency via Constrained Deferred Acceptance.\
  Denis Derkach and Alexander Nesterov.\
  [[Bibtex](<link>)][[Paper](<link>)]

---

### Development Team

Contributors: [Denis Derkach](https://github.com/Dpbt) and Alexander Nesterov.






























