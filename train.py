import numpy as np, torch, random
from config import HP, Explore, EnvCfg, TB
from env import RobotEnv
from agent import Agent
from logging_tb import TBLogger
from metrics import EpisodeStats

def set_seed(seed = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main():
    set_seed(42)
    tb = TBLogger(TB.logdir_root)
    tb.add_notes("seed=42 | env=Robot5x5 | commit=<hash>")
    tb.add_hp("learning_rate", HP.lr, 0)

    env = RobotEnv(EnvCfg())
    agent = Agent(
        hp = HP(), ex = Explore(),
        in_dim = 11, out_dim = 3,
        tb_logger = tb
    )

    env.reset()
    stats = EpisodeStats(ma_window = 100)

    while True:
        state_old = agent.get_state(env)
        final_move, move_idx = agent.get_action(state_old)
        done, reward = env.step(final_move)

        state_new = agent.get_state(env)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        # train.py
        stats.record_step(reward, move_idx)

        if done:
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            success = 1.0 if reward > 0 else 0.0
            stats.close_episode(success)

            # --- TB scalars ---
            tb.scalar("metrics/success", success, agent.n_games)
            tb.scalar("metrics/success_ma100", stats.success_ma, agent.n_games)
            tb.scalar("train/episode_length", stats.step_count, agent.n_games)

            # action “histogram” as scalars (readable)
            tb.scalar("actions/straight", stats.action_counts[0], agent.n_games)
            tb.scalar("actions/right", stats.action_counts[1], agent.n_games)
            tb.scalar("actions/left", stats.action_counts[2], agent.n_games)

            stats.reset_episode()

if __name__ == "__main__":
    main()
