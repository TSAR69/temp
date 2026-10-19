import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from step4_train_ppo import CustomRewardLander

pygame.init()

env = CustomRewardLander(gym.make("LunarLander-v3", render_mode="human"))
model = PPO.load("ppo_rocket_pilot_live")

obs, info = env.reset()
done = False

print("Running default evaluation simulation...")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    env.render()
    done = terminated or truncated

env.close()
