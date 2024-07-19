import numpy as np

class PolicyIteration:
    def __init__(self, env, discount_factor=0.9, theta=0.0001, max_iterations=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.policy = np.ones([env.nS, env.nA]) / env.nA
        self.value_function = np.zeros(env.nS)
        print("PolicyIteration initialized")

    def policy_evaluation(self):
        print("Starting policy evaluation")
        iteration = 0
        while iteration < self.max_iterations:
            delta = 0
            for state in range(self.env.nS):
                v = 0
                for action, action_prob in enumerate(self.policy[state]):
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        v += action_prob * prob * (reward + self.discount_factor * self.value_function[next_state])
                delta = max(delta, np.abs(v - self.value_function[state]))
                self.value_function[state] = v
            iteration += 1
            print(f"Iteration: {iteration}, Delta: {delta}")
            if delta < self.theta:
                break
        if iteration == self.max_iterations:
            print("Policy evaluation did not converge within the maximum number of iterations")

    def policy_improvement(self):
        print("Starting policy improvement")
        policy_stable = True
        for state in range(self.env.nS):
            chosen_action = np.argmax(self.policy[state])
            action_values = np.zeros(self.env.nA)
            for action in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    action_values[action] += prob * (reward + self.discount_factor * self.value_function[next_state])
            best_action = np.argmax(action_values)
            if chosen_action != best_action:
                policy_stable = False
            self.policy[state] = np.eye(self.env.nA)[best_action]
        print("Policy improvement complete, policy stable:", policy_stable)
        return policy_stable

    def iterate(self):
        print("Starting policy iteration")
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"Policy Iteration Step: {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break
        if iteration == self.max_iterations:
            print("Policy iteration did not converge within the maximum number of iterations")
        return self.policy, self.value_function
