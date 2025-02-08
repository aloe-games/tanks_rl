import gymnasium as gym

from tanks_grid_world_env import TanksGridWorldEnv

gym.register(
    id="aloe_games_env/TanksGridWorld-v0",
    entry_point=TanksGridWorldEnv,
)

env = gym.make("aloe_games_env/TanksGridWorld-v0", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
