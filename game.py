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
        self.fps = 200

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
        self._update_ui()
        pygame.display.flip()
        pygame.time.delay(80)

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
        # 1) handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # pace BEFORE the move so first step after reset isn't a burst
        self.clock.tick(self.fps)

        # --- positions BEFORE move ---
        robot_pos_before = (self.robot.y // self.bs, self.robot.x // self.bs)
        goal_pos = (self.goal.y // self.bs, self.goal.x // self.bs)
        distance_before = manhattan(robot_pos_before, goal_pos)

        # 2) move (turn + step)
        self._move(action)

        # 3) terminal checks (so we can render the final frame once)
        reached_goal = (self.robot == self.goal)
        out_of_bounds = (self.robot.x <= 0 or self.robot.x >= self.w or
                         self.robot.y <= 0 or self.robot.y >= self.h)
        r = self.robot.y // self.bs
        c = self.robot.x // self.bs
        in_bounds = (0 <= r < self.cfg.rows and 0 <= c < self.cfg.cols)
        stepped_in_hole = (in_bounds and self.grid[r][c] == "Hole")

        if reached_goal or out_of_bounds or stepped_in_hole:
            # draw last frame so we SEE why it ended
            self._update_ui()
            cx, cy = int(self.robot.x), int(self.robot.y)
            radius = self.bs // 3

            # only clamp if truly OOB; otherwise keep actual center
            if out_of_bounds:
                pad = radius + 2
                draw_x = min(max(cx, pad), self.w - pad)
                draw_y = min(max(cy, pad), self.h - pad)
            else:
                draw_x, draw_y = cx, cy

            # Fill body, then outline with success/fail color
            pygame.draw.circle(self.display, Colors.WHITE, (draw_x, draw_y), radius, 0)
            color = (0, 255, 0) if reached_goal else (255, 0, 0)
            pygame.draw.circle(self.display, color, (draw_x, draw_y), radius, 6)

            pygame.display.flip()
            pygame.time.delay(120)

            if reached_goal:
                return True, 30.0
            else:
                return True, -15.0

        # 4) non-terminal: distance shaping
        robot_pos_after = (self.robot.y // self.bs, self.robot.x // self.bs)
        distance_after = manhattan(robot_pos_after, goal_pos)

        shaping_factor = 0.2
        reward = shaping_factor * (distance_before - distance_after)
        reward -= 0.01

        # 5) render & speed
        self._update_ui()

        return False, reward

    def _update_ui(self):
        self.display.fill(Colors.BLACK)
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                rect = pygame.Rect(c * self.bs, r * self.bs, self.bs, self.bs)
                pygame.draw.rect(self.display, self.cfg.color_map[self.grid[r][c]], rect)
                pygame.draw.rect(self.display, Colors.BLACK, rect, width=2)

        # draw robot body + facing line
        cx, cy = int(self.robot.x), int(self.robot.y)
        rad = self.bs // 3

        pygame.draw.circle(self.display, Colors.WHITE, (cx, cy), rad, 2)

        line_len = self.bs // 2  # longer arrow
        if self.direction == Direction.RIGHT:
            end = (cx + line_len, cy)
        elif self.direction == Direction.LEFT:
            end = (cx - line_len, cy)
        elif self.direction == Direction.UP:
            end = (cx, cy - line_len)
        elif self.direction == Direction.DOWN:
            end = (cx, cy + line_len)

        pygame.draw.line(self.display, (255, 0, 0), (cx, cy), end, 5)
        pygame.display.flip()