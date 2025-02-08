from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TanksGridWorldEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.MultiDiscrete([[8] * 13 for _ in range(13)])
        self.action_space = spaces.Discrete(8)

        self._map = self._default_map()

    def _default_map(self):
        # 0: empty
        # 1: brick
        # 2: forest
        # 3: metal
        # 4: water
        return np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 2, 3, 3, 2, 1, 1, 0, 1],
            [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 1],
            [0, 1, 1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1],
            [0, 0, 1, 1, 3, 0, 1, 0, 3, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 2, 1, 1, 1, 3, 3, 1, 1, 1, 1, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
        ], dtype=int)

    def _get_obs(self):
        return self._map

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._map = self._default_map()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
