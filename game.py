import random
from collections import namedtuple, deque
from enum import Enum

import numpy as np
import pygame

pygame.init()
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x,y')
BLOCK_SIZE, ROWS, COLS = 100, 5, 5

#colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)
RED = (255, 0, 0)

# field color/tile type
COLOR_MAP = {
    "End": YELLOW,  # goal
    "Start": BLUE,  # start
    "Normal": GREY, # rest
    "Hole": RED,    # hole
}
field_counts = {
    "Normal": 20,
    "Hole": 3,
    "Start": 1,
    "End": 1,
}


def get_random_fields(rows, cols, counts):
    total = rows * cols
    if sum(counts.values()) != total:
        raise ValueError(f"Counts sum to {sum(counts.values())}, but grid needs {total}")

    # build pool
    pool = []
    for name, n in counts.items():
        pool.extend([name] * n)

    random.shuffle(pool)  # shuffle in- place
    # reshape to rows * cols
    grid_reshaped = [pool[i*cols:(i+1)*cols] for i in range(rows)]
    return grid_reshaped

def find_first(grid, value):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == value:
                return r, c
    return None

def manhattan(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def has_path(grid, start_rc, end_rc):
    """Return True if End is reachable from Start using 4-neighborhood, avoiding 'Hole' tiles."""
    R, C = len(grid), len(grid[0])
    sr, sc = start_rc
    er, ec = end_rc
    if grid[sr][sc] == "Hole" or grid[er][ec] == "Hole":
        return False

    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]  # up,down,left,right

    while q:
        r, c = q.popleft()
        if (r, c) == (er, ec):
            return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C \
               and (nr, nc) not in seen \
               and grid[nr][nc] != "Hole":
                seen.add((nr, nc))
                q.append((nr, nc))
    return False

class RobotGame:
    def __init__(self):
        # Make window fit the grid 1:1
        self.w = COLS*BLOCK_SIZE
        self.h = ROWS*BLOCK_SIZE

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Robot Game 5x5")
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT

        # --- generate grid with constraints: distance >= 4 and path exists ---
        self.grid = None
        self.goal = None
        for _ in range(10_000):
            grid = get_random_fields(ROWS, COLS, field_counts)
            s = find_first(grid, "Start")
            e = find_first(grid, "End")
            if manhattan(s, e) < 4:
                continue
            if not has_path(grid, s, e):
                continue

            # valid grid!
            self.grid = grid
            start_row, start_col = s
            break
        self.goal = s
        self.robot = Point(
            start_col * BLOCK_SIZE + BLOCK_SIZE // 2,
            start_row * BLOCK_SIZE + BLOCK_SIZE // 2
        )

    def is_collision(self):
        # hits boundary
        if self.robot.x > self.w or self.robot.x < 0 or self.robot.y > self.h  or self.robot.y < 0:
            return True
        # goes in hole
        if self.grid[self.robot.y // BLOCK_SIZE][self.robot.x // BLOCK_SIZE] == "Hole":
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
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.robot = Point(x, y)

    def play_step(self, action):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

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
            reward = 10
            return game_over, reward

        # 3.self. update ui
        self._update_ui()
        self.clock.tick(30)
        # 4. return game over
        return game_over, reward

    def _update_ui(self):
        self.display.fill(BLACK)
        for r in range(ROWS):
            for c in range(COLS):
                rect = pygame.Rect(c * BLOCK_SIZE, r * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, COLOR_MAP[self.grid[r][c]], rect)
                pygame.draw.rect(self.display, BLACK, rect, width=2)
        # draw robot circle
        cx, cy = int(self.robot.x), int(self.robot.y)
        pygame.draw.circle(self.display, (255, 255, 255), (cx, cy), BLOCK_SIZE // 3, 2)
        pygame.display.flip()

