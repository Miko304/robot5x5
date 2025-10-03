import numpy as np, torch, random
from collections import deque
from model import Linear_QNet, QTrainer
from memory import ReplayBuffer
from config import HP, Explore


class Agent:
    def __init__(self, hp: HP, ex: Explore, in_dim : int, out_dim: int, tb_logger = None):
        self.n_games = 0
        self.hp, self.ex = hp, ex
        self.epsilon = ex.eps_start
        self.memory = ReplayBuffer(hp.max_memory)
        self.model = Linear_QNet(in_dim, 128, out_dim)
        self.trainer = QTrainer(self.model, lr = hp.lr, gamma = hp.gamma, tb_writer = (tb_logger.writer if tb_logger else None))
        self.tb = tb_logger
        self.last_moves = deque(maxlen = 4)

    def _eps(self):
        self.epsilon = max(self.ex.eps_end, self.ex.eps_start * (self.ex.eps_decay ** self.n_games))
        return self.epsilon

    def get_state(self, env) -> np.ndarray:
        robot = env.robot
        Point = type(env.robot)
        point_l = Point(robot.x - 100, robot.y)
        point_r = Point(robot.x + 100, robot.y)
        point_u = Point(robot.x, robot.y - 100)
        point_d = Point(robot.x, robot.y + 100)

        dir_l = env.direction.name == "LEFT"
        dir_r = env.direction.name == "RIGHT"
        dir_u = env.direction.name == "UP"
        dir_d = env.direction.name == "DOWN"

        state = [
            # Danger straight
            (dir_r and env.is_collision(point_r)) or
            (dir_l and env.is_collision(point_l)) or
            (dir_u and env.is_collision(point_u)) or
            (dir_d and env.is_collision(point_d)),

            # Danger right
            (dir_u and env.is_collision(point_r)) or
            (dir_d and env.is_collision(point_l)) or
            (dir_l and env.is_collision(point_u)) or
            (dir_r and env.is_collision(point_d)),

            # Danger left
            (dir_d and env.is_collision(point_r)) or
            (dir_u and env.is_collision(point_l)) or
            (dir_r and env.is_collision(point_u)) or
            (dir_l and env.is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # End location
            env.goal.x < env.robot.x, # goal left
            env.goal.x > env.robot.x, # goal right
            env.goal.y < env.robot.y, # goal up
            env.goal.y > env.robot.y, # goal down
        ]

        return np.array(state, dtype = int)

    def remember(self, *args):
        self.memory.push(*args)

    def train_long_memory(self):
        batch = self.memory.sample(self.hp.batch_size)
        if not batch:
            return
        states, actions, rewards, next_states, dones = zip(*batch)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        if self.tb: self.tb.scalar("loss/td", loss, self.trainer.train_steps)

    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        if self.tb:
            self.tb.scalar("loss/td", loss, self.trainer.train_steps)

    def get_action(self, state):
        eps = self._eps()
        out_dim = self.model.linear2.out_features
        final_move = [0] * out_dim
        if random.randint(0, 200) < (eps*200):
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                q = self.model(torch.tensor(state, dtype=torch.float))
            q = q.cpu().numpy()
            winners = np.flatnonzero(q >= q.max() - 1e-6)
            move = int(np.random.choice(winners))

        final_move[move] = 1
        self.last_moves.append(move)
        if self.tb: self.tb.scalar("hp/epsilon", eps, self.n_games)
        return final_move, move

