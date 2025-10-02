import os

import torch
import random
import numpy as np
from collections import deque

from game import RobotGame, Direction, Point
from model import Linear_QNet, QTrainer

from torch.utils.tensorboard import SummaryWriter
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
IMAGE_EVERY_N_EPISODES = 10 # every 10 episodes screenshot of board

class Agent:
    def __init__(self, tb_writer = None):
        self.n_games = 0
        self.epsilon = 0
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 0.995
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        self.last_moves = deque(maxlen=4)
        self.tb_writer = tb_writer
        self.last_100_success = deque(maxlen=100)

    def get_state(self, game):
        robot = game.robot
        point_l = Point(robot.x - 100, robot.y)
        point_r = Point(robot.x + 100, robot.y)
        point_u = Point(robot.x, robot.y - 100)
        point_d = Point(robot.x, robot.y + 100)


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # End location
            game.goal.x < game.robot.x, # goal left
            game.goal.x > game.robot.x, # goal right
            game.goal.y < game.robot.y, # goal up
            game.goal.y > game.robot.y, # goal down
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/td", loss, self.trainer.train_steps)

    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss/td", loss, self.trainer.train_steps)

    def get_action(self, state):
        self.epsilon = max(self.eps_end, self.eps_start * (self.eps_decay ** self.n_games))
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            q = prediction.cpu().numpy()
            winners = np.flatnonzero(q >= q.max() - 1e-6)  # near-max indices
            move = int(np.random.choice(winners))  # pick randomly among ties
            final_move[move] = 1

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("hp/epsilon", max(self.epsilon, 0), self.n_games)

        chosen = np.argmax(final_move)
        self.last_moves.append(chosen)

        # if last 4 moves were all "right" (index 1) or all "left" (index 2), break the loop once
        if len(self.last_moves) == self.last_moves.maxlen:
            if all(m == 1 for m in self.last_moves):  # always turning right
                final_move = [1, 0, 0]  # go straight once
            elif all(m == 2 for m in self.last_moves):  # always turning left
                final_move = [1, 0, 0]  # go straight once

        return final_move

def train():
    # init Writer
    run_name = time.strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=os.path.join("runs_robot5x5", run_name))

    # Run-Notes
    tb_writer.add_text("run/notes", "seed=42 | env=Robot5x5-v1 | commit=<your_commit_hash>", 0)
    tb_writer.add_text("run/meta", f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", 0)
    tb_writer.add_scalar("hp/learning_rate", LR, 0)

    agent = Agent()
    game = RobotGame()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    episode_steps = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        done, reward  = game.play_step(final_move)

        episode_steps += 1

        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # success: reward >0 (goal reached) otherwise = 0
            success = 1.0 if reward > 0 else 0.0

            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Scalars per Episode
            agent.last_100_success.append(success)
            sr_ma100 = np.mean(agent.last_100_success) if agent.last_100_success else 0.0

            tb_writer.add_scalar("metrics/success", success, agent.n_games)  # 0 or 1
            tb_writer.add_scalar("metrics/success_ma100", sr_ma100, agent.n_games)  # smoothed line

            tb_writer.add_scalar("train/episode_length", episode_steps, agent.n_games)
            # Log action histogram for the episode

            # reset Episode Stats
            episode_steps = 0



if __name__ == '__main__':
    train()