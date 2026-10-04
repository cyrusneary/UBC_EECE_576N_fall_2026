from typing import List, Protocol
import numpy as np
import math
import random
import matplotlib.pyplot as plt

############ ARM DEFINITION ############
class BernoulliArm:
    """A class representing an individual Bernoulli arm with success probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p
    def sample(self) -> float:
        return 1.0 if random.random() < self.p else 0.0
    def mean(self) -> float:
        return self.p

########### Environment definition ###########
class BanditEnv:
    """A class representing a multi-armed bandit environment with multiple arms."""

    def __init__(self, arms: List[BernoulliArm]):
        self.arms = arms

    def pull(self, i: int) -> float:
        return self.arms[i].sample()

    def best_mean(self) -> float:
        return max(a.mean() for a in self.arms)

    def num_arms(self) -> int:
        return len(self.arms)
    
############ Policy protocol (follow this structure) ############
class Policy(Protocol):
    def __init__(self, num_arms: int):
        ...

    def select_arm(self) -> int:
        ...

    def update(self, arm: int, reward: float) -> None:
        ...

############ Bandit simulation and plotting functions ############
def run_bandit(
    env: BanditEnv,
    policy: Policy,
    horizon: int,
):
    rewards = np.zeros(horizon, dtype=float)
    regret = np.zeros(horizon, dtype=float)
    cumulative_regret = 0.0
    best = env.best_mean()

    for t in range(horizon):
        a = policy.select_arm()
        r = env.pull(a)
        policy.update(a, r)
        rewards[t] = r
        regret[t] = best - env.arms[a].mean()

    cumulative_regret = np.cumsum(regret)
    return cumulative_regret

def plot_bandit_estimates(env, policy, title="Bandit arm estimates with CIs"):
    """
    env: BanditEnv with .arms and each arm has .mean()
    policy: has .values (estimates), .radii (CI radii), .counts (pull counts), .num_arms
    """
    k = policy.num_arms
    x = np.arange(k)

    compute_radii(policy)  # ensure radii are up-to-date

    true_means = np.array([env.arms[i].mean() for i in range(k)])
    est = np.array(policy.values, dtype=float)
    rad = np.array(policy.radii, dtype=float)
    cnt = np.array(policy.counts, dtype=int)

    # Hide CIs for unpulled arms
    yerr = rad.copy()
    yerr[cnt == 0] = np.nan  # avoid giant/infinite bars for unseen arms

    plt.figure(figsize=(8,5))

    # True means as 'x' markers
    plt.scatter(x, true_means, marker='x', s=80, label="True mean")

    # Estimated means with error bars (estimate ± radius)
    plt.errorbar(
        x, est, yerr=yerr, fmt='o', capsize=4, label="Estimate ± CI radius"
    )

    # Add the current time step policy.t as text
    if hasattr(policy, 't'):
        plt.text(0.95, 0.95, f"t = {policy.t}", horizontalalignment='right',
                 verticalalignment='top', transform=plt.gca().transAxes)

    # Cosmetic touches
    plt.ylim(-1, 2)
    plt.xticks(x, [f"Arm {i}" for i in x])
    plt.xlabel("Arm")
    plt.ylabel("Reward mean")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

############ Bandit problem a) SOLN: CI and Uniform Policy ############

def compute_radii(policy: Policy) -> None:
    """
    Compute the confidence interval radii for each arm in the policy.
    This function modifies the policy in place, updating its `radii` attribute.
    """
    assert policy.num_arms > 0, "Policy must have num_arms > 0"
    assert policy.counts is not None and len(policy.counts) == policy.num_arms, "Policy must have counts array of length num_arms"
    assert policy.t > 0, "Policy time step t must be > 0 to compute radii"
    assert policy.radii is not None and len(policy.radii) == policy.num_arms, "Policy must have radii array of length num_arms"

    #TODO: Implement the computation of the confidence interval radii
    pass

class UniformPolicy:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.t = 0
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.radii = np.zeros(self.num_arms, dtype=float)

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

########### Bandit problem c) SOLN: UCB1 ###########
class UCB1:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.t = 0
        self.radii = np.zeros(self.num_arms, dtype=float)        

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

########### Bandit problem d) SOLN: Various Other Policy types ##########
class UniformExploration:

    def __init__(self, num_arms: int, exploration_rounds: int = 1):
        self.num_arms = num_arms
        self.exploration_rounds = exploration_rounds
        
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.t = 0
        self.radii = np.zeros(self.num_arms, dtype=float)
        self.best_arm_found = None

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

class EpsilonGreedy:
    def __init__(self, num_arms: int, epsilon: float = 0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.t = 0
        self.radii = np.zeros(self.num_arms, dtype=float)        

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

class SuccessiveElimination:

    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)
        self.active_arms = set(range(self.num_arms))
        self.active_arms_in_cycle = set(self.active_arms)
        self.t = 0
        self.radii = np.zeros(self.num_arms, dtype=float)

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

class ThompsonSamplingBernoulli:
    def __init__(self, num_arms: int, alpha0: float = 1.0, beta0: float = 1.0):
        self.num_arms = num_arms
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.counts = np.zeros(self.num_arms, dtype=int)
        self.values = np.zeros(self.num_arms, dtype=float)  # mean rewards
        self.t = 0
        self.radii = np.zeros(self.num_arms, dtype=float)

        self.alpha = np.full(self.num_arms, self.alpha0, dtype=float)  # successes + 1
        self.beta = np.full(self.num_arms, self.beta0, dtype=float)    # failures + 1

    def select_arm(self) -> int:
        #TODO
        pass

    def update(self, arm: int, reward: float) -> None:
        #TODO
        pass

if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    
    # ############ Bandits question a) ############
    #TODO

    ############ Bandits question b) ############
    #TODO

    ############ Bandits question c) ############
    #TODO

    ############ Bandits question d) ############
    #TODO