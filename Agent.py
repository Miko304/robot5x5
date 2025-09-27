import torch
import random
import numpy as np
from collections import deque
from game import RobotGame, Direction

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # TODO: Model and Trainer

    def get_state(self, game):
        robot = (game.robot_row , game.robot_col)
        point_l = robot(+100, -100)


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight

        ]
