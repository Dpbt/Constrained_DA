from utils import generate_random_profiles, generate_school_capacities, generate_k_restricted_preferences, calculate_utility
from algorithm import k_boston_algorithm, k_gs_algorithm


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

    if algorithm == 'boston':
        k = num_schools
        preferences = generate_k_restricted_preferences(profiles, k)
        assignments, unassigned_students = k_boston_algorithm(num_students=num_students,
                                                              num_schools=num_schools,
                                                              preferences=preferences,
                                                              capacities=capacities,
                                                              k=k)

    elif algorithm == 'gs':
        for k in range(1, num_schools + 1):
            k_gs_algorithm()

    else:
        raise ValueError('Algorithm must be either "boston" or "gs"')

    utilities = calculate_utility(num_students=num_students, assignments=assignments, profiles=profiles)

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
