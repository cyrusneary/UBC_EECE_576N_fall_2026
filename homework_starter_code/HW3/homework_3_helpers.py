import numpy as np

def optimal_q_bellman_operator(mdp, Q):
    """
    One sweep of the optimal Bellman operator over Q.
    new_Q(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q(s', a')
    """
    new_Q = np.zeros_like(Q)
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            r_sa = mdp._reward_sa(s, a)
            p_sas = mdp.transition_probabilities(s, a)  # dict {s_next: prob}
            boot = 0.0
            for s_next, p in p_sas.items():
                boot += p * np.max(Q[s_next])
            new_Q[s, a] = r_sa + mdp.discount * boot
    return new_Q

def q_iteration(mdp, max_iter=1000, tol=1e-6):
    """
    Q-iteration: repeatedly apply the optimal Q Bellman operator.
    Returns:
      Q: action-value table of shape (nS, nA)
      iters: number of iterations performed
    """
    Q = np.zeros((mdp.nS, mdp.nA))
    for i in range(max_iter):
        new_Q = optimal_q_bellman_operator(mdp, Q)
        if np.max(np.abs(new_Q - Q)) < tol:
            print(f"Converged after {i+1} iterations.")
            return new_Q, i
        Q = new_Q
    return Q, max_iter

def get_optimal_Q_V_and_policy(mdp, max_iter=1000, tol=1e-6):
    """
    Compute the optimal Q-function, V-function, and policy for the given MDP.
    Returns:
      Q_optimal: optimal action-value function of shape (nS, nA)
      V_optimal: optimal state-value function of shape (nS,)
      policy_optimal: optimal policy of shape (nS, nA) (deterministic)
    """
    Q_optimal, _ = q_iteration(mdp, max_iter=max_iter, tol=tol)
    V_optimal = np.max(Q_optimal, axis=1)
    policy_optimal = np.zeros((mdp.nS, mdp.nA))
    best_actions = np.argmax(Q_optimal, axis=1)
    for s in range(mdp.nS):
        policy_optimal[s, best_actions[s]] = 1.0
    return Q_optimal, V_optimal, policy_optimal