# from numpy import ndarray
import numpy as np

from utils import (generate_random_profiles, generate_school_capacities, generate_k_restricted_preferences,
                   calculate_utility, calculate_utilities_from_prob)
from algorithm import k_boston_algorithm, k_gs_algorithm, manipulation_algorithm, algorithm_sampler


def run_experiment(algorithm: str,
                   num_students: int,
                   num_schools: int,
                   num_repeat: int,
                   epsilon: float,
                   num_manipulations: int):
    if num_students < num_schools:
        raise ValueError("Number of students cannot be less than the number of schools.")

    profiles = generate_random_profiles(num_students=num_students, num_schools=num_schools)
    capacities = generate_school_capacities(num_students=num_students, num_schools=num_schools)
    # print(profiles)
    # print(capacities)

    # На данный момент считается, что boston только для k = num_schools
    if algorithm == 'boston':
        k = num_schools
        preferences = generate_k_restricted_preferences(profiles, k)
        # assignments, unassigned_students = k_boston_algorithm(num_students=num_students,
        #                                                       num_schools=num_schools,
        #                                                       preferences=preferences,
        #                                                       capacities=capacities,
        #                                                       k=k)

        probabilities = algorithm_sampler(algorithm='k_gs',
                                          num_students=num_students,
                                          num_schools=num_schools,
                                          preferences=preferences,
                                          capacities=capacities,
                                          k=k,
                                          num_repeat=num_repeat)

        utilities = calculate_utilities_from_prob(num_students=num_students,
                                                  num_schools=num_schools,
                                                  probabilities=probabilities,
                                                  profiles=profiles)

    elif algorithm == 'gs':
        for k in range(1, num_schools + 1):
            preferences, manipulators = manipulation_algorithm(algorithm=algorithm,
                                                               num_students=num_students,
                                                               num_schools=num_schools,
                                                               profiles=profiles,
                                                               capacities=capacities,
                                                               k=k,
                                                               epsilon=epsilon,
                                                               num_manipulations=num_manipulations)

            probabilities = algorithm_sampler(algorithm='k_gs',
                                              num_students=num_students,
                                              num_schools=num_schools,
                                              preferences=preferences,
                                              capacities=capacities,
                                              k=k,
                                              num_repeat=num_repeat)

            # print("true probabilities", probabilities, sep='\n')

            utilities = calculate_utilities_from_prob(num_students=num_students,
                                                      num_schools=num_schools,
                                                      probabilities=probabilities,
                                                      profiles=profiles)

            # print("true utilities", utilities, sep='\n')
    else:
        raise ValueError('Algorithm must be either "boston" or "gs"')

    # utilities = calculate_utility(num_students=num_students, assignments=assignments, profiles=profiles)

    return assignments, unassigned_students, utilities


def run_experiment_k(algorithm: str,
                   num_students: int,
                   num_schools: int,
                   profiles: np.ndarray,
                   capacities: np.ndarray,
                   num_repeat_sampler: int,
                   # num_repeat_for_k: int,
                   epsilon: float,
                   num_manipulations: int,
                   k: int):

    # На данный момент считается, что boston только для k = num_schools
    if algorithm == 'boston':
        k = num_schools

    if algorithm == 'boston':
        preferences = generate_k_restricted_preferences(profiles, k)
    elif algorithm == 'gs':
        # Возможно, тут надо еще и это много раз повторять
        preferences, manipulators = manipulation_algorithm(algorithm=algorithm,
                                                           num_students=num_students,
                                                           num_schools=num_schools,
                                                           profiles=profiles,
                                                           capacities=capacities,
                                                           k=k,
                                                           epsilon=epsilon,
                                                           num_manipulations=num_manipulations)
    else:
        raise ValueError('Algorithm must be either "boston" or "gs"')

    probabilities = algorithm_sampler(algorithm=algorithm,
                                      num_students=num_students,
                                      num_schools=num_schools,
                                      preferences=preferences,
                                      capacities=capacities,
                                      k=k,
                                      num_repeat=num_repeat_sampler)

    utilities = calculate_utilities_from_prob(num_students=num_students,
                                                      num_schools=num_schools,
                                                      probabilities=probabilities,
                                                      profiles=profiles)

    return assignments, unassigned_students, utilities


def massive_run():
    # Принимает список экспериментов, запускает их параллельно, формирует pd.df
    pass



if __name__ == '__main__':
    assignments, unassigned_students, utilities = run_experiment(algorithm='boston',
                                                                 num_students=10,
                                                                 num_schools=4,
                                                                 num_repeat=100,
                                                                 epsilon=0.1,
                                                                 num_manipulations=0)

    # print(assignments)
    # print(unassigned_students)
    # print(utilities)
    # print(preferences)
