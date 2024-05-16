from policy.ucb_policy import UCBPolicy
from stable_baselines3 import DQN
import numpy as np

class DQN_UCB(DQN):
    def __init__(self, policy, env, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=32, ucb_c=1.5, *args, **kwargs):
        self.ucb_c = ucb_c  # Store ucb_c as an instance variable
        self.policy = UCBPolicy(env.observation_space, env.action_space, ucb_c=self.ucb_c)
        super(DQN_UCB, self).__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, *args, **kwargs)

    def _setup_model(self):
        super(DQN_UCB, self)._setup_model()


class CustomDQN(DQN):
    """
    Custom DQN model to include UCB in action selection.
    """
    def __init__(self, *args, ucb_c=2.0, **kwargs):
        super(CustomDQN, self).__init__(*args, **kwargs)
        self.ucb_c = ucb_c
        self.action_counts = np.zeros(self.env.action_space.n)

    def _predict(self, observation, deterministic=False):
        q_values = super(CustomDQN, self).predict(observation, deterministic=True)  # Always use the current Q estimates
        if deterministic:
            action = np.argmax(q_values)
        else:
            total_counts = np.sum(self.action_counts)
            ucb_values = q_values + self.ucb_c * np.sqrt(np.log(total_counts + 1) / (self.action_counts + 1))
            action = np.argmax(ucb_values)
            self.action_counts[action] += 1
        return action