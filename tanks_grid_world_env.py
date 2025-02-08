from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Cell(Enum):
    EMPTY = 0
    BRICK = 1
    FOREST = 2
    METAL = 3
    WATER = 4
    TANK = 5
    ENEMY = 6
    BULLET = 7


class TanksGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.height, self.width = self.default_map().shape
        self.observation_space = spaces.MultiDiscrete([[8] * self.width for _ in range(self.height)])
        self.action_space = spaces.Discrete(8)
        self.render_mode = render_mode

        self.map = self.default_map()
        self.tank = self.default_tank()
        self.enemy = self.default_enemy()
        self.bullets = self.default_bullets()

        self.window = None
        self.clock = None
        self.size = 50

    def default_map(self):
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

    def default_tank(self):
        return np.array([12, 4], dtype=int)

    def default_enemy(self):
        return np.array([0, 6], dtype=int)

    def default_bullets(self):
        return [np.array([0, 2])]

    def get_obs(self):
        obs = self.map.copy()
        # forest can hide tank, enemy or bullet
        if obs[self.tank[0], self.tank[1]] != Cell.FOREST.value:
            obs[self.tank[0], self.tank[1]] = Cell.TANK.value
        if obs[self.enemy[0], self.enemy[1]] != Cell.FOREST.value:
            obs[self.enemy[0], self.enemy[1]] = Cell.ENEMY.value
        for bullet in self.bullets:
            if obs[bullet[0], bullet[1]] != Cell.FOREST.value:
                obs[bullet[0], bullet[1]] = Cell.BULLET.value
        return obs

    def get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.map = self.default_map()
        self.tank = self.default_tank()
        self.enemy = self.default_enemy()
        self.bullets = self.default_bullets()

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 1

        self.tank[0] -= 1

        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def cell_color(self, cell):
        colors = {
            Cell.EMPTY.value: pygame.Color(0, 0, 0),
            Cell.BRICK.value: pygame.Color(205, 133, 63),
            Cell.FOREST.value: pygame.Color(154, 205, 50),
            Cell.METAL.value: pygame.Color(255, 255, 255),
            Cell.WATER.value: pygame.Color(100, 149, 237),
            Cell.TANK.value: pygame.Color(255, 215, 0),
            Cell.ENEMY.value: pygame.Color(255, 99, 71),
            Cell.BULLET.value: pygame.Color(169, 169, 169)
        }
        return colors[cell]

    def draw_block(self, color, position):
        pygame.draw.rect(self.window, color, pygame.Rect(position[1] * self.size, position[0] * self.size, self.size, self.size))

    def render(self):
        if self.window is None:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((self.size * self.width, self.size * self.height))
            pygame.display.set_caption("Tanks Grid World")

        obs = self.get_obs()
        for i in range(self.height):
            for j in range(self.width):
                self.draw_block(self.cell_color(obs[i, j]), (i, j))

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
