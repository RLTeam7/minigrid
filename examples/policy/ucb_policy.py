import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
class UCBPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, ucb_c=1.5):
        self.ucb_c = ucb_c
        super(UCBPolicy, self).__init__(observation_space, action_space)
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

# custom policy using UCB
class customPolicy(DQNPolicy):
    def __init__(self, *args, ucb_c=2.0, **kwargs):
        self.ucb_c = ucb_c
        super(customPolicy, self).__init__(*args, **kwargs)
        self.action_counts = np.zeros(self.env.action_space.n)
        
    def predict(self, observation, state=None, mask=None, deterministic=False):
        q_values = super(customPolicy, self).predict(observation, state, mask, deterministic=True)[0]  # Always use the current Q estimates
        if deterministic:
            action = np.argmax(q_values)
            print('not ok')
        else:
            total_counts = np.sum(self.action_counts)
            ucb_values = q_values + self.ucb_c * np.sqrt(np.log(total_counts + 1) / (self.action_counts + 1))
            action = np.argmax(ucb_values)
            self.action_counts[action] += 1
            print('ok')
        return action, state        

    # def _predict(self, observation, deterministic=False):
    #     q_values = super(customPolicy, self).predict(observation, deterministic=True)  # Always use the current Q estimates
    #     if deterministic:
    #         action = np.argmax(q_values)
    #     else:
    #         total_counts = np.sum(self.action_counts)
    #         ucb_values = q_values + self.ucb_c * np.sqrt(np.log(total_counts + 1) / (self.action_counts + 1))
    #         action = np.argmax(ucb_values)
    #         self.action_counts[action] += 1
    #     return action