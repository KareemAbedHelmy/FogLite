# Green Fog RL

Reinforcement learning-based task scheduler for energy-efficient fog–cloud computing.

## Structure

- `env_fog.py` — Python discrete-event simulator of fog + cloud environment (Gym-style).
- `baselines.py` — heuristic schedulers (Round Robin, Least Loaded, Energy-Min).
- `train_dqn.py` — training script for DQN (or other RL methods).
- `config/nodes_config.py` — node and VM configuration inspired by Azure-style fog + cloud topology.
- `notebooks/` — exploratory experiments / plots.
- `tests/` — simple tests to sanity-check the environment.
