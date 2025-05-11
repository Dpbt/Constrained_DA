# Constrained_DA

## Project Overview

## Project Overview

This repository was developed as part of the paper “Balancing Incentives and Efficiency via Constrained Deferred Acceptance” 
and provides a set of tools for modeling and analyzing various matching mechanisms, including the constrained 
Deferred Acceptance mechanism (Gale-Shapley), the Boston mechanism, and the Chinese parallel mechanism. 
The core of the project is a system for running simulations of these mechanisms in large markets. 
Also in the project you can find clean implementations of these mechanisms and use them separately. 

In addition to the system of simulations and algorithms, the toolkit includes such utilities as:

- A search engine for finding (symmetric) Nash equilibria within these mechanisms
- Tools for generating and analyzing small exact examples of the mechanisms' performance
- Additional functions to support system operation and analysis

## Principle of the simulation system

The main system of the project is started from the `run_experiment.py` file and works according to the following scheme: 

The system simulates school enrollment mechanisms, student behavior (i.e., manipulations) for the $DA^k$ mechanism 
with different list lengths. The Boston mechanism with a fair list (without manipulations) of equal length to the number 
of schools is also tested. The main goal in doing so is to find out what is the optimal list length in terms 
of the average utility that students receive and whether the $DA^ k$ mechanism with optimal k is better 
than the unconstrained Boston mechanism. Each simulation proceeds as follows.

First, fix the number of students n (parameter num\_students) and the number of schools m (parameter num\_schools). 
Also fix the percentage of sophisticated student (i.e., students allowed to manipulate, parameter sophisticated\_students\_ratio), 
the maximum number of manipulations (i.e., changes in the list) available to each sophisticated student 
(parameter num\_manipulations\_ratio), and the minimum acceptable profit from manipulations (parameter min\_profit).
We also record the technical parameters of the experiment, such as the number of cardinal preference profile generations 
(parameter num\_profiles), the number of school capacity generations (parameter num\_capacities), and the number of runs 
of the mechanism on the final student spikes (parameter num\_repeats\_mechanism). Subsequently, all data obtained 
for each generation are averaged between these repetitions.

Having fixed all the above parameters, num\_capacities of different sets of school capacities are generated so that 
the sum of capacities of all schools is equal to the number of students num\_student (in case you have not set 
the capacity parameter yourself).

Then, for all fixed parameters and for each set of school capacities, num\_profiles of the same ordinal but different 
cardinal preferences for all students and schools are generated. The cardinal preferences for each student are taken 
from an m-dimensional simplex (i.e., the cardinal preferences for each student sum to 1).

The experiment is then run a given number of times for $DA^k$ with $k = 1, ..., m$, i.e., for all possible list lengths. 
For $DA^m$ and the Boston mechanism, no manipulation is done (each student submits a fair full list), and the mechanisms 
are implemented directly by simply computing the distribution of students across schools and averaging the results over 
num\_repeats\_mechanism repetitions. In the case of $DA^k$ with $k < m$, the mechanisms are manipulable and are handled 
through a manipulation process. The details of its operation should be seen in the paper. The result of the manipulation 
process is a list of schools of length $k$ from each of the students.  On these lists, the $DA^k$ mechanism is run 
num\_repeats\_mechanism once, and then all results from the runs are averaged. 

You can understand the principles of the system's operation in more detail by studying the corresponding block in our paper, 
as well as using the documentation for the functions in this repository.

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
- **`data_out/`**: Directory for storing experiment results (now contains the simulation results files for the paper)
  - **`analysis/`**: Directory for graphs and tables, contains some tables from the paper
  - **`technical/`**: A directory for saving intermediate results during simulations. Can be useful when simulations are stopped / error before full completion
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
   cd Constrained_DA
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv constrained_da_venv
   source constrained_da_venv/bin/activate  # On Windows: constrained_da_venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
---

## Usage

### Running Experiments

The main entry point for running experiments is `run_experiment.py`. 
You can customize and run experiments by providing parameter dictionaries in this file.


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
The results are saved as CSV files in the `data_out` directory. Each file contains detailed results of the experiment, including:
- Parameters of the specific simulation
- Students' mean utilities
- Manipulation statistics
- Percentages of unassigned students
- Metrics related to a specific algorithm
- List length ranking within a specific set of parameters
- Other statistics

---

## Running using docker

1. Install Docker and Docker Compose:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo systemctl enable --now docker
   ```

2. Transfer the Project Files to the Server.

3. Build and Start the Project:
   ```bash
   sudo docker-compose up --build -d
   ```

4. Check the Logs:
   ```bash
   sudo docker-compose logs -f
   ```

5. Stop and Remove the Containers:
   ```bash
   sudo docker-compose down
   ```
   
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






























