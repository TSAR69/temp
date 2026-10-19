import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class CustomRewardLagerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_potential = None

    def calculate_potential(self, obs):
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs
        distance_score = -100.0 * float(np.sqrt(x**2 + y**2))
        speed_score = -100.0 * float(np.sqrt(vx**2 + vy**2))
        angle_score = -100.0 * float(abs(angle)) - 20.0 * float(abs(angular_vel))
        leg_bonus = 10.0 * float(left_leg) + 10.0 * float(right_leg)
        return distance_score + speed_score + angle_score + leg_bonus

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        self.prev_potential = self.calculate_potential(obs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32)
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs

        current_potential = self.calculate_potential(obs)
        custom_reward = current_potential - self.prev_potential
        self.prev_potential = current_potential

        if action == 2:
            custom_reward -= 0.30
        elif action in [1, 3]:
            custom_reward -= 0.03

        custom_reward -= 0.05

        if terminated:
            is_crashed = self.env.unwrapped.game_over or abs(x) > 1.0
            is_safe_landing = (
                not is_crashed 
                and left_leg == 1.0 
                and right_leg == 1.0 
                and abs(vx) < 0.2 
                and abs(vy) < 0.2
            )
            if is_crashed:
                custom_reward -= 100.0
            elif is_safe_landing:
                custom_reward += 100.0

        return obs, float(custom_reward), terminated, truncated, info

def make_custom_env():
    return CustomRewardLagerWrapper(gym.make("LunarLander-v3"))

if __name__ == "__main__":
    model_path = "ppo_rocket_final.zip"
    
    if not os.path.exists(model_path):
        print("🚀 Initializing parallel training (Fast Headless Mode)...")
        train_env = make_vec_env(make_custom_env, n_envs=4)

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
        )

        print("Training PPO rocket pilot (takes ~2 minutes)...")
        model.learn(total_timesteps=200_000)
        model.save("ppo_rocket_final")
        print("Training complete! Model saved.")
    else:
        print("Found existing model! Loading...")
        model = PPO.load("ppo_rocket_final")

    print("\n🎬 Launching simulation: modi aayaaa with Abhijeet Dipke 🐶 tracking...")
    
    pygame.init()
    eval_env = CustomRewardLagerWrapper(gym.make("LunarLander-v3", render_mode="human"))
    obs, info = eval_env.reset()
    done = False

    font = pygame.font.SysFont("Arial", 18, bold=True)
    pygame.display.set_caption("modi aayaaa")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs
        eval_env.render()
        
        screen = pygame.display.get_surface()
        if screen is not None:
            screen_width, screen_height = screen.get_size()
            pixel_x = int((x + 1.0) * (screen_width / 2.0))
            pixel_y = int(screen_height - (y * (screen_height / 1.5))) - 40
            
            text_surface = font.render("Abhijeet Dipke 🐶", True, (0, 255, 128))
            text_rect = text_surface.get_rect(center=(pixel_x, pixel_y))
            
            screen.blit(text_surface, text_rect)
            pygame.display.flip()

        done = terminated or truncated

    eval_env.close()
