import itertools

import numpy as np


def find_nash_equilibrium(
        results: list[tuple[np.ndarray, np.ndarray]],
        profiles: np.ndarray = None,
        symmetric: bool = False
) -> list[tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]]:
    """
    Identifies Nash equilibria in strategic games with optional symmetry constraints.

    Analyzes game outcomes to find strategy profiles where no player/group can
    unilaterally improve their payoff. Supports both standard and symmetric equilibria
    through utility-based player grouping.

    Parameters:
        results (list): Game outcomes as [(strategy_profile, utilities)] where:
            - strategy_profile: np.ndarray of player choices (shape: (players, strategies))
            - utilities: np.ndarray of corresponding payoffs (shape: (players,))
        profiles (np.ndarray): Matrix of utility profiles (shape: (players, num_utilities))
            Required when symmetric=True
        symmetric (bool): Restrict to equilibria where identical-utility players
            use identical strategies

    Returns:
        list: Nash equilibria as [(strategy_profile, utilities)] where:

            - strategy_profile: Tuple of player strategy tuples

            - utilities: Corresponding payoff tuple

    Raises:
        ValueError: When symmetric=True and profiles=None
    """
    utility_dict = {}
    # Create utility dictionary with immutable types
    for pref, utils in results:
        strategy_profile = tuple(
            tuple(int(x) for x in player_strategy) for player_strategy in pref
        )
        utilities = tuple(float(u) for u in utils)
        utility_dict[strategy_profile] = utilities

    # Determine game parameters
    num_players = len(next(iter(utility_dict.keys())))
    players_strategies = []

    # Build strategy spaces
    for player_idx in range(num_players):
        strategies = {tuple(int(x) for x in pref[player_idx]) for pref, _ in results}
        players_strategies.append(list(strategies))

    # Player grouping logic
    utils_groups = {}
    if symmetric:
        if profiles is None:
            raise ValueError("profiles parameter is required when symmetric=True")

        for player_idx, prefs in enumerate(profiles):
            key = tuple(int(x) for x in prefs)
            utils_groups.setdefault(key, []).append(player_idx)

    def is_nash_equilibrium(strategy_profile):
        current_utilities = utility_dict[strategy_profile]

        for player_idx in range(num_players):
            for candidate_strategy in players_strategies[player_idx]:
                if candidate_strategy == strategy_profile[player_idx]:
                    continue

                new_profile = list(strategy_profile)
                new_profile[player_idx] = candidate_strategy
                new_profile = tuple(new_profile)

                if new_profile not in utility_dict:
                    continue

                if utility_dict[new_profile][player_idx] > current_utilities[player_idx]:
                    return False
        return True

    # Profile generation and validation
    nash_equilibria = []
    for strategy_profile in itertools.product(*players_strategies):
        if strategy_profile not in utility_dict:
            continue

        # Symmetry validation
        valid_profile = True
        if symmetric:
            for group_indices in utils_groups.values():
                if len({strategy_profile[i] for i in group_indices}) != 1:
                    valid_profile = False
                    break

        if valid_profile and is_nash_equilibrium(strategy_profile):
            nash_equilibria.append((strategy_profile, utility_dict[strategy_profile]))

    # Console reporting
    if not nash_equilibria:
        print("No Nash equilibria found.")
    else:
        print(f"Found {len(nash_equilibria)} Nash equilibria ({'symmetric' if symmetric else 'all types'}):")
        for profile, utils in nash_equilibria:
            players_str = ", ".join(f"Player {i + 1}: {strat}" for i, strat in enumerate(profile))
            utils_str = ", ".join(f"{u:.2f}" for u in utils)
            print(f"{players_str} | Utilities: {utils_str}")

    return nash_equilibria
