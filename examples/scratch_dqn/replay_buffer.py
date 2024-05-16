from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, sample_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)