from dataclasses import dataclass, field

@dataclass(frozen=True)
class HP:
    lr: float = 0.001
    gamma: float = 0.9
    batch_size: int = 1000
    max_memory: int = 100_000

@dataclass(frozen=True)
class Explore:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

class Colors:
    BLACK = (0, 0, 0)
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    GREY = (100, 100, 100)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)

@dataclass(frozen=True)
class EnvCfg:
    block_size: int = 100
    rows: int = 5
    cols: int = 5

    # how many tiles of each type the grid should contain
    field_counts: dict = field(default_factory=lambda: {
        "Normal": 20,
        "Hole": 3,
        "Start": 1,
        "End": 1,
    })

    # tile -> display color
    color_map: dict = field(default_factory=lambda: {
        "End": Colors.YELLOW,
        "Start": Colors.BLUE,
        "Normal": Colors.GREY,
        "Hole": Colors.RED,
    })

@dataclass(frozen=True)
class TB:
    logdir_root: str = "runs_robot5x5"

