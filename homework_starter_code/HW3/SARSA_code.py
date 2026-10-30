import numpy as np
from gridworld_gym_env import GridworldEnv

import matplotlib.pyplot as plt

from homework_3_helpers import get_optimal_Q_V_and_policy

def sarsa(
    env: GridworldEnv,
    num_episodes: int,
    alpha: float,
    gamma: float = 1.0,
    epsilon: float = 0.1,
):
    """
    SARSA algorithm to learn optimal action-value function Q and policy.
    Args:
        env: Gym environment
        num_episodes: number of episodes to sample
        alpha: step-size parameter
        gamma: discount factor
        epsilon: probability of choosing a random action (epsilon-greedy)
    Returns:
        Q_list: list of Q arrays (numpy arrays of dimension (env.nS, env.nA)) at the start of each episode.
        N_visit_list: list of N_visit arrays (numpy arrays of dimension (env.nS)) at the start of each episode.
    """

    Q_list = [] # to store a copy of Q at the start of each episode
    N_visit_list = [] # to store a copy of N_visit at the start of each episode
    Q = np.zeros((env.nS, env.nA), dtype=np.float64)
    N_visit = np.zeros(env.nS, dtype=np.int32)

    for _ in range(num_episodes):
        # TODO: Implement the main body of the SARSA algorithm.
        pass

    return Q_list, N_visit_list

if __name__ == "__main__":

    # Environment parameters
    gamma = 0.9
    epsilon = 0.1

    # Construct the MDP instance
    env = GridworldEnv(
        height=6,
        width=9,
        init=(2, 0),
        goal=(0, 8),
        sink=(5, 8),
        wall=[(1, 2), (2, 2), (3,2), (4,5), (0,7), (1,7), (2,7)],
        reward_goal=+1.0,
        reward_sink=-1.0,
        step_cost=-0.1,
        slip_p=0.05,  # 5% chance to slip,
        discount=gamma,
    )
    
    # Compute optimal Q, V, and policy via Q-iteration for comparison with your RL-based results.
    Q_optimal, V_optimal, policy_optimal = get_optimal_Q_V_and_policy(env.mdp, max_iter=10000, tol=1e-6)
    
    # Sanity check: visualize the MDP, optimal value function, and optimal policy
    env.mdp.plot_grid()
    env.mdp.plot_values(V_optimal, annotate=True)
    env.mdp.plot_policy(policy_optimal)

    # SARSA experiment
    alpha_list = [0.01, 0.05, 0.15] # different learning rates to try
    num_episodes = 10000

    # TODO: Implement answers to homework questions.