import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env_fog import FogEnv
class FogGymWrapper(gym.Env):
    """
    Wraps your FogEnv into a Gymnasium-compatible environment
    so Stable-Baselines3 DQN can train on it.

    Requirements:
    - observation_space
    - action_space
    - reset()
    - step()
    """

    metadata = {"render_modes": []}

    def __init__(self, fog_env: FogEnv):
        super().__init__()
        self.fog_env = fog_env
        # Observation space is a vector of floats
        # FogEnv builds state of shape: num_nodes + 2
        dummy_state = self.fog_env.reset()
        obs_dim = dummy_state.shape[0]

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        # Action space = choose node index
        self.action_space = spaces.Discrete(self.fog_env.num_nodes)

    def reset(self, seed=None, options=None):
        obs = self.fog_env.reset()
        return obs, {}

    def step(self, action):
        next_state, reward, done, info = self.fog_env.step(int(action))
        if next_state is None:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_state, reward, done, False, info
