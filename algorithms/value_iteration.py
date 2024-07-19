import numpy as np

class ValueIteration:
    def __init__(self, env, discount_factor=0.9, theta=0.0001, max_iterations=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.value_function = np.zeros(env.nS)
        self.policy = np.zeros([env.nS, env.nA])

    def iterate(self):
        print("Starting value iteration")
        for i in range(self.max_iterations):
            delta = 0
            for state in range(self.env.nS):
                v = self.value_function[state]
                action_values = np.zeros(self.env.nA)
                for action in range(self.env.nA):
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        action_values[action] += prob * (reward + self.discount_factor * self.value_function[next_state])
                best_action_value = np.max(action_values)
                self.value_function[state] = best_action_value
                delta = max(delta, np.abs(v - best_action_value))
            print(f"Iteration: {i+1}, Delta: {delta}")
            if delta < self.theta:
                break
        for state in range(self.env.nS):
            action_values = np.zeros(self.env.nA)
            for action in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    action_values[action] += prob * (reward + self.discount_factor * self.value_function[next_state])
            best_action = np.argmax(action_values)
            self.policy[state] = np.eye(self.env.nA)[best_action]
        return self.policy, self.value_function
