from collections import namedtuple
from enum import Enum
import numpy as np
import pygame
from config import EnvCfg, Colors
from grid_utils import get_random_fields, find_first, manhattan, has_path

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x,y')

class RobotGame:
    def __init__(self, cfg: EnvCfg = EnvCfg()):
        self.cfg = cfg
        self.bs = cfg.block_size
        self.w = self.cfg.cols * cfg.block_size
        self.h = self.cfg.rows * cfg.block_size

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Robot Game 5x5")
        self.clock = pygame.time.Clock()

        self.direction: Direction = Direction.RIGHT
        self.grid = None
        self.robot: Point = Point(0,0)
        self.goal: Point = Point(0,0)

        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT

        # --- generate grid with constraints: distance >= 4 and path exists ---
        self.grid = None
        self.goal = None
        start_row = start_col = end_row = end_col = None
        for _ in range(10_000):
            grid = get_random_fields(self.cfg.rows, self.cfg.cols, self.cfg.field_counts)
            s = find_first(grid, "Start")
            e = find_first(grid, "End")
            if manhattan(s, e) < 4:
                continue
            if not has_path(grid, s, e):
                continue

            # valid grid!
            self.grid = grid
            start_row, start_col = s
            end_row, end_col = e
            break
        if self.grid is None:
            raise RuntimeError("Failed to generate a valid grid after 10,000 tries.")

        self.goal = Point(
            end_col * self.bs + self.bs // 2,
            end_row * self.bs + self.bs // 2,
        )
        self.robot = Point(
            start_col * self.bs + self.bs // 2,
            start_row * self.bs + self.bs // 2,
        )

    def is_collision(self, pt = None):
        if pt is None:
            pt = self.robot
        # hits boundary
        if pt.x > self.w or pt.x < 0 or pt.y > self.h  or pt.y < 0:
            return True
        # goes in hole
        if self.grid[pt.y // self.bs][pt.x // self.bs] == "Hole":
            return True
        return False

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0,0,1]
            next_idx = (idx -1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir
        x = self.robot.x
        y = self.robot.y
        if self.direction == Direction.RIGHT:
            x += self.bs
        elif self.direction == Direction.LEFT:
            x -= self.bs
        elif self.direction == Direction.UP:
            y -= self.bs
        elif self.direction == Direction.DOWN:
            y += self.bs

        self.robot = Point(x, y)

    def play_step(self, action):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Position before move
        robot_pos_before = (self.robot.y // self.bs, self.robot.x // self.bs)
        goal_pos = (self.goal.y // self.bs, self.goal.x // self.bs)
        distance_before = manhattan(robot_pos_before, goal_pos)

        self._move(action)

        # 2. check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return game_over, reward

        if self.robot == self.goal:
            game_over = True
            reward = 15
            return game_over, reward

        # Position after move
        robot_pos_after = (self.robot.y // self.bs, self.robot.x // self.bs)
        distance_after = manhattan(robot_pos_after, goal_pos)

        # shaping factor
        shaping_factor = 0.2
        reward = shaping_factor * (distance_before - distance_after)

        # 3.self. update ui
        self._update_ui()
        self.clock.tick(10)
        # 4. return game over
        return game_over, reward

    def _update_ui(self):
        self.display.fill(Colors.BLACK)
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                rect = pygame.Rect(c * self.bs, r * self.bs, self.bs, self.bs)
                pygame.draw.rect(self.display, self.cfg.color_map[self.grid[r][c]], rect)
                pygame.draw.rect(self.display, Colors.BLACK, rect, width=2)
        # draw robot circle
        cx, cy = int(self.robot.x), int(self.robot.y)
        r = self.bs // 3

        pygame.draw.circle(self.display, Colors.WHITE, (cx, cy), r, 2)

        line_len = self.bs // 2  # longer arrow
        if self.direction == Direction.RIGHT:
            end = (cx + line_len, cy)
        elif self.direction == Direction.LEFT:
            end = (cx - line_len, cy)
        elif self.direction == Direction.UP:
            end = (cx, cy - line_len)
        elif self.direction == Direction.DOWN:
            end = (cx, cy + line_len)

        pygame.draw.line(self.display, (255, 0, 0), (cx, cy), end, 5)  # red, thicker

        pygame.display.flip()