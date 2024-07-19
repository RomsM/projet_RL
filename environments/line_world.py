class LineWorld:
    def __init__(self, length=5):
        self.length = length
        self.nS = length
        self.nA = 2
        self.P = self._create_transition_probabilities()
        print(f"LineWorld initialized with length: {self.length}")

    def _create_transition_probabilities(self):
        P = {state: {action: [] for action in range(self.nA)} for state in range(self.nS)}
        for state in range(self.nS):
            if state > 0:
                P[state][0] = [(1.0, state - 1, 1, False)]  # Reward is now 1
            else:
                P[state][0] = [(1.0, state, 0, True)]
            if state < self.length - 1:
                P[state][1] = [(1.0, state + 1, 1, False)]  # Reward is now 1
            else:
                P[state][1] = [(1.0, state, 0, True)]
        return P

    def reset(self):
        self.state = 0
        print("Environment reset")
        return self.state

    def step(self, action):
        prob, next_state, reward, done = self.P[self.state][action][0]
        self.state = next_state
        print(f"Action: {action}, New state: {self.state}, Reward: {reward}, Done: {done}")
        return next_state, reward, done, {}

    def render(self):
        line = ['-'] * self.length
        line[self.state] = 'A'
        print(''.join(line))
