import numpy as np
from collections import defaultdict

class QLearning:
    def __init__(self, env, discount_factor=1.0, alpha=0.5, epsilon=0.1, num_episodes=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = defaultdict(lambda: np.zeros(env.nA))

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def iterate(self):
        policy = self.make_epsilon_greedy_policy(self.Q, self.epsilon, self.env.nA)
        for i_episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            for t in range(100):
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = self.env.step(action)
                best_next_action = np.argmax(self.Q[next_state])
                self.Q[state][action] += self.alpha * (reward + self.discount_factor * self.Q[next_state][best_next_action] - self.Q[state][action])
                if done:
                    break
                state = next_state

        policy = defaultdict(lambda: np.zeros(self.env.nA))
        for state, actions in self.Q.items():
            best_action = np.argmax(actions)
            policy[state] = np.eye(self.env.nA)[best_action]

        return policy, self.Q
