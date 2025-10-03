Robot 5×5
A small reinforcement learning project where a robot learns to navigate a 5×5 randomized grid world.
The robot can only move forward, turn left, or turn right, and must reach the goal tile while avoiding holes and out-of-bounds.

Training is logged with TensorBoard for easy monitoring.

Start training -> python train.py
Start tensorBoard -> tensorboard --logdir=runs_robot5x5

What it does
-Generates random 5×5 maps with start, goal, normal tiles, and holes.
-The robot learns via Deep Q-Learning (DQN):
  -State → Neural network → Action (straight / left / right).
  -Rewards: +15 goal, –10 fail, shaping bonus for moving closer to goal.
-Logs success rate, episode length, loss, gradients, parameters, and actions to TensorBoard.
-Visualizes the grid and robot with pygame.
