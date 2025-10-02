from dataclasses import dataclass

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

@dataclass(frozen=True)
class EnvCfg:
    block_size: int = 100
    rows: int = 5
    cols: int = 5

@dataclass(frozen=True)
class TB:
    logdir_root: str = "runs_robot5x5"

