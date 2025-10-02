from collections import deque
import numpy as np

class EpisodeStats:
    def __init__(self, ma_window: int = 100):
        self.step_count = 0
        self.reward_sum = 0.0
        self.success_list = deque(maxlen=ma_window)
        self.action_counts = [0,0,0]

    def record_step(self, reward: float, action_idx: int):
        self.step_count += 1
        self.reward_sum += reward
        self.action_counts[action_idx] += 1

    def close_episode(self, success: float):
        self.success_list.append(success)

    @property
    def success_ma(self) -> float:
        return float(np.mean(self.success_list)) if self.success_list else 0.0

    def reset_episode(self):
        self.step_count = 0
        self.reward_sum = 0.0
        self.action_counts = [0,0,0]