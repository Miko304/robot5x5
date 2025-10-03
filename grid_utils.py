# grid_utils.py
import random
from collections import deque
from typing import List, Tuple, Optional, Dict

Grid = List[List[str]]
RC = Tuple[int, int]

def get_random_fields(rows: int, cols: int, counts: Dict[str, int]) -> Grid:
    total = rows * cols
    if sum(counts.values()) != total:
        raise ValueError(f"Counts sum to {sum(counts.values())}, but grid needs {total}")

    pool: List[str] = []
    for name, n in counts.items():
        pool.extend([name] * n)

    random.shuffle(pool)
    return [pool[i*cols:(i+1)*cols] for i in range(rows)]

def find_first(grid: Grid, value: str) -> Optional[RC]:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == value:
                return (r, c)
    return None

def manhattan(a: RC, b: RC) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def has_path(grid: Grid, start_rc: RC, end_rc: RC) -> bool:
    """
    True if `end_rc` is reachable from `start_rc` using 4-neighborhood,
    avoiding 'Hole' tiles.
    """
    R, C = len(grid), len(grid[0])
    sr, sc = start_rc
    er, ec = end_rc

    if grid[sr][sc] == "Hole" or grid[er][ec] == "Hole":
        return False

    q: deque[RC] = deque([(sr, sc)])
    seen = {(sr, sc)}
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]

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
