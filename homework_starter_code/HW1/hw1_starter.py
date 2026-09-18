from gridworld_code import GridworldMDP
import numpy as np

# TODO: Initialize gridworld
# mdp = GridworldMDP(...)

# TODO: Create a uniform policy

####### Policy iteration question
# TODO: Implement the policy iteration algorithm

def policy_iteration(mdp, init_policy):
    pass

# Visualize the value of a uniform random policy
V = policy_iteration(mdp, pi_uniform)
mdp.plot_values(V, annotate=True)

####### Value iteration question
# TODO: Implement the value iteration algorithm

def value_iteration(mdp, max_iter=1000, tol=1e-6):
    pass


V = value_iteration(mdp, tol=0.1)
mdp.plot_values(V, annotate=True)