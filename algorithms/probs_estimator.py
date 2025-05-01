import numpy as np

from utils import generate_statistic


def k_gs_algorithm_prob_individual(num_schools: int,
                                   preferences: np.ndarray,
                                   capacities: np.ndarray,
                                   k: int,
                                   student: int) -> list[float]:
    """
        Estimates the probabilities of assigning a student in a constrained Deferred Acceptance mechanism
        based on the statistics of the distribution of schools by place on student preference lists.

        Parameters:
            num_schools (int): Total number of available schools
            preferences (np.ndarray): Student preference matrix (shape: (num_students, k))
                                      Each row represents a student's school preferences per round
            capacities (np.ndarray): School capacities (shape: (num_schools,))
            k (int): Length of preference lists (k <= num_schools)
            student (int): Target student index for probability estimation

        Returns:
            list[float]: Probability distribution over schools (indexed 0 to num_schools-1)
        """
    statistic = generate_statistic(num_schools=num_schools, preferences=preferences, k=k)
    probabilities = [0 for _ in range(num_schools)]

    curr_sum = 0

    for curr_preference in range(k):
        curr_school = preferences[student, curr_preference]
        curr_prob = 1 - curr_sum

        num_competitors = 0

        for curr_step in range(k):
            avg_capacities = (np.sum(capacities[:curr_school]) / curr_school) * curr_step if curr_school > 0 else 0
            curr_stats = np.sum(statistic[:curr_step, :curr_school])

            if curr_stats > 0:
                prob_assigned = avg_capacities / curr_stats
            else:
                prob_assigned = 0
            prob_unassigned = 1 - prob_assigned
            num_competitors += prob_unassigned * statistic[curr_step, curr_school]

        final_prob = curr_prob * capacities[curr_school] / num_competitors
        probabilities[curr_school] = final_prob
        curr_sum += final_prob

    return probabilities