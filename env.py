from typing import Tuple, Optional
from game import RobotGame, Point
from config import EnvCfg

class RobotEnv:
    def __init__(self, cfg: EnvCfg = EnvCfg()):
        self.cfg = cfg
        self.game = RobotGame()

    def reset(self) -> None:
        self.game.reset()

    def step(self, action) -> Tuple[bool, float]:
        done, reward = self.game.play_step(action)
        return done, reward

    # convenience passthroughs
    @property
    def robot(self) -> Point:
        return self.game.robot

    @property
    def goal(self) -> Point:
        return self.game.goal

    @property
    def direction(self):
        return self.game.direction

    def is_collision(self, point: Optional[Point] = None) -> bool:
        return self.game.is_collision(point)
