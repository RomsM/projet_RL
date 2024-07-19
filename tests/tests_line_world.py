import sys
import os
import numpy as np
from collections import defaultdict

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.on_policy_mc import OnPolicyFirstVisitMC
from algorithms.off_policy_mc import OffPolicyMC
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.dyna_q import DynaQ

def test_algorithm(algorithm_class, algorithm_name, num_runs=10, **kwargs):
    rewards = []
    iterations = []
    
    for _ in range(num_runs):
        env = LineWorld(length=5)
        agent = algorithm_class(env, **kwargs)
        policy, value_function = agent.iterate()
        
        total_reward = 0
        state = env.reset()
        for _ in range(env.length):
            action = np.argmax(policy[state])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        rewards.append(total_reward)
        if hasattr(agent, 'num_iterations'):
            iterations.append(agent.num_iterations)
        else:
            iterations.append(len(agent.Q))  # Rough estimate if num_iterations is not defined

    avg_reward = np.mean(rewards)
    avg_iterations = np.mean(iterations)
    
    print(f"Results for {algorithm_name}:")
    print(f"Average Reward: {avg_reward}")
    print(f"Average Iterations: {avg_iterations}")
    
    return avg_reward, avg_iterations

if __name__ == "__main__":
    algorithms = [
        (PolicyIteration, "Policy Iteration", {"discount_factor": 0.9, "theta": 0.0001, "max_iterations": 1000}),
        (ValueIteration, "Value Iteration", {"discount_factor": 0.9, "theta": 0.0001, "max_iterations": 1000}),
        (MonteCarloES, "Monte Carlo ES", {"discount_factor": 0.9, "epsilon": 0.1, "num_episodes": 1000}),
        (OnPolicyFirstVisitMC, "On-policy First-Visit MC", {"discount_factor": 0.9, "epsilon": 0.1, "num_episodes": 1000}),
        (OffPolicyMC, "Off-policy MC", {"discount_factor": 0.9, "epsilon": 0.1, "num_episodes": 1000}),
        (SARSA, "SARSA", {"discount_factor": 0.9, "alpha": 0.5, "epsilon": 0.1, "num_episodes": 1000}),
        (QLearning, "Q-Learning", {"discount_factor": 0.9, "alpha": 0.5, "epsilon": 0.1, "num_episodes": 1000}),
        (DynaQ, "Dyna-Q", {"discount_factor": 0.9, "alpha": 0.5, "epsilon": 0.1, "planning_steps": 50, "num_episodes": 1000})
    ]

    results = []
    for algorithm_class, algorithm_name, kwargs in algorithms:
        print(f"\nTesting {algorithm_name} on Line World")
        avg_reward, avg_iterations = test_algorithm(algorithm_class, algorithm_name, num_runs=10, **kwargs)
        results.append((algorithm_name, avg_reward, avg_iterations))
    
    # Analyzing the results
    print("\nComparison of Algorithms on Line World:")
    for name, reward, iterations in results:
        print(f"{name}: Average Reward = {reward}, Average Iterations = {iterations}")
