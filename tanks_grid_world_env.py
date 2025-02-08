from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class TanksGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.MultiDiscrete([[8] * 13 for _ in range(13)])
        self.action_space = spaces.Discrete(8)
        self.render_mode = render_mode

        self._map = self._default_map()

        self._window = None
        self._clock = None

    def _default_map(self):
        return np.array([
            [0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
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
            [0, 0, 0, 1, 5, 1, 0, 1, 0, 0, 1, 0, 0]
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

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _cell_color(self, cell):
        # 0: empty
        # 1: brick
        # 2: forest
        # 3: metal
        # 4: water
        # 5: tank
        # 6: bullet
        # 7: enemy
        colors = {
            0: pygame.Color(0, 0, 0),
            1: pygame.Color(206, 76, 10),
            2: pygame.Color(132, 211, 17),
            3: pygame.Color(255, 255, 255),
            4: pygame.Color(33, 56, 239),
            5: pygame.Color(255, 154, 57),
            6: pygame.Color(189, 190, 189),
            7: pygame.Color(255, 57, 57)
        }
        return colors[cell]

    def render(self):
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((650, 650))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("Tanks Grid World")

        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                cell = self._map[i, j]
                pygame.draw.rect(self._window, self._cell_color(cell), pygame.Rect(j * 50, i * 50, 50, 50))

        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata['render_fps'])

    def close(self):
        if self._window is not None:
            pygame.quit()
