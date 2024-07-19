import numpy as np
from collections import defaultdict

class OffPolicyMC:
    def __init__(self, env, discount_factor=1.0, epsilon=0.1, num_episodes=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = defaultdict(lambda: np.zeros(env.nA))
        self.C = defaultdict(lambda: np.zeros(env.nA))

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def iterate(self):
        target_policy = defaultdict(lambda: np.zeros(self.env.nA))
        behavior_policy = self.make_epsilon_greedy_policy(self.Q, self.epsilon, self.env.nA)

        for i_episode in range(1, self.num_episodes + 1):
            episode = []
            state = self.env.reset()
            for t in range(100):
                probs = behavior_policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            G = 0.0
            W = 1.0
            for t in range(len(episode))[::-1]:
                state, action, reward = episode[t]
                G = self.discount_factor * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                best_action = np.argmax(self.Q[state])
                target_policy[state] = np.eye(self.env.nA)[best_action]
                if action != best_action:
                    break
                W *= 1.0 / behavior_policy(state)[action]

        return target_policy, self.Q
