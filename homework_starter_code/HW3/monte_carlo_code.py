import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from homework_3_helpers import get_optimal_Q_V_and_policy
from gridworld_gym_env import GridworldEnv

######### MC Prediction (First-Visit) code #########
def generate_episode(env, policy, max_steps: int = 1000):
    """
    Roll out one episode using the given policy.
    Returns:
        states:  list of observations (tuples) per time step
        actions: list of actions taken per time step
        rewards: list of rewards per time step
    """
    # TODO: Implement generate_episode function
    pass

def mc_prediction_first_visit(
    env,
    policy,
    num_episodes: int = 500000,
    gamma: float = 1.0,
):
    """
    Monte-Carlo prediction (first-visit) to estimate V^pi.
    Args:
        env: Gym environment
        policy: function mapping state -> action
        num_episodes: number of episodes to sample
        gamma: discount factor
    Returns:
        V_list: list of V arrays (numpy arrays of dimension (env.nS)) at the start of each episode.
        N_list: list of N arrays (numpy arrays of dimension (env.nS)) at the start of each episode.
    """
    # TODO: Implement mc_prediction_first_visit function
    pass

if __name__ == "__main__":

    # Environment parameters
    gamma = 0.9

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

    ############ Monte-Carlo Prediction experiment ###########  
    # TODO: Implement the code to answer the homework questions.