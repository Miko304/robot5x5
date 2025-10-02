from collections import deque
from typing import Deque, Tuple, List
import random

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)