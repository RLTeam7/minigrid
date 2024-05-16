import numpy as np
from stable_baselines3.common.policies import BasePolicy

class UCBPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, ucb_c=1.5):
        super(UCBPolicy, self).__init__(observation_space, action_space)
        self.ucb_c = ucb_c
        self.action_counts = np.zeros(self.action_space.n, dtype=np.float32)
        self.total_counts = 0

    def _predict(self, observation, deterministic=False):
        q_values = super(UCBPolicy, self).predict(observation, deterministic=True)
        
        if not deterministic:
            ucb_values = q_values + self.ucb_c * np.sqrt(np.log(self.total_counts + 1) / (self.action_counts + 1))
            action = np.argmax(ucb_values)
        else:
            action = np.argmax(q_values)

        # Update counters
        self.action_counts[action] += 1
        self.total_counts += 1

        return action