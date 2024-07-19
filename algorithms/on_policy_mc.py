import numpy as np
from collections import defaultdict

class OnPolicyFirstVisitMC:
    def __init__(self, env, discount_factor=1.0, epsilon=0.1, num_episodes=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = defaultdict(lambda: np.zeros(env.nA))
        self.policy = defaultdict(lambda: np.zeros(env.nA))

    def make_epsilon_greedy_policy(self):
        def policy_fn(observation):
            A = np.ones(self.env.nA, dtype=float) * self.epsilon / self.env.nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn

    def iterate(self):
        policy = self.make_epsilon_greedy_policy()
        returns_sum = defaultdict(lambda: np.zeros(self.env.nA))
        returns_count = defaultdict(lambda: np.zeros(self.env.nA))

        for i_episode in range(1, self.num_episodes + 1):
            episode = []
            state = self.env.reset()
            for t in range(100):
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            sa_in_episode = set([(x[0], x[1]) for x in episode])
            for state, action in sa_in_episode:
                first_occurrence_idx = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
                G = sum([x[2] * (self.discount_factor ** i) for i, x in enumerate(episode[first_occurrence_idx:])])
                returns_sum[state][action] += G
                returns_count[state][action] += 1.0
                self.Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                best_action = np.argmax(self.Q[state])
                self.policy[state] = np.eye(self.env.nA)[best_action]

        return self.policy, self.Q
