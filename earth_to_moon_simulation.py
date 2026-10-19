import os
import math
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
        print("🚀 Training PPO Lunar Lander model in background...")
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
        model.learn(total_timesteps=150_000)
        model.save("ppo_rocket_final")
        print("Training complete!")
    else:
        model = PPO.load("ppo_rocket_final")

    pygame.init()
    WIDTH, HEIGHT = 900, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Earth to Moon Transfer Simulation")
    clock = pygame.time.Clock()

    EARTH_POS = (WIDTH // 2, HEIGHT // 2)
    MOON_DISTANCE = 280

    class Spacecraft:
        def __init__(self, x, y, vx, vy):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.trajectory = []

        def update(self, moon_pos):
            dx_e = EARTH_POS[0] - self.x
            dy_e = EARTH_POS[1] - self.y
            dist_e = math.sqrt(dx_e**2 + dy_e**2) + 1e-5
            
            dx_m = moon_pos[0] - self.x
            dy_m = moon_pos[1] - self.y
            dist_m = math.sqrt(dx_m**2 + dy_m**2) + 1e-5

            acc_e_x = (dx_e / dist_e) * (8000 / (dist_e**2))
            acc_e_y = (dy_e / dist_e) * (8000 / (dist_e**2))
            
            acc_m_x = (dx_m / dist_m) * (600 / (dist_m**2))
            acc_m_y = (dy_m / dist_m) * (600 / (dist_m**2))

            self.vx += acc_e_x + acc_m_x
            self.vy += acc_e_y + acc_m_y
            self.x += self.vx
            self.y += self.vy

            self.trajectory.append((int(self.x), int(self.y)))

    rocket = Spacecraft(EARTH_POS[0], EARTH_POS[1] - 35, vx=3.2, vy=0)
    moon_angle = 0.0
    orbiting = True
    font = pygame.font.SysFont("Arial", 20)

    print("🌍 Simulating Earth Liftoff and Trans-Lunar Cruise...")
    while orbiting:
        screen.fill((10, 10, 25))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        moon_angle += 0.003
        moon_x = EARTH_POS[0] + math.cos(moon_angle) * MOON_DISTANCE
        moon_y = EARTH_POS[1] + math.sin(moon_angle) * MOON_DISTANCE
        current_moon_pos = (moon_x, moon_y)

        rocket.update(current_moon_pos)

        pygame.draw.circle(screen, (30, 144, 255), EARTH_POS, 24)
        pygame.draw.circle(screen, (200, 200, 200), (int(moon_x), int(moon_y)), 10)

        if len(rocket.trajectory) > 1:
            pygame.draw.aalines(screen, (255, 255, 0), False, rocket.trajectory)

        pygame.draw.circle(screen, (255, 255, 255), (int(rocket.x), int(rocket.y)), 4)

        txt = font.render("Phase 1: Trans-Lunar Cruise to the Moon...", True, (255, 255, 255))
        screen.blit(txt, (20, 20))

        pygame.display.flip()
        clock.tick(60)

        dist_to_moon = math.sqrt((rocket.x - moon_x)**2 + (rocket.y - moon_y)**2)
        if dist_to_moon < 45 or len(rocket.trajectory) > 600:
            orbiting = False

    pygame.quit()
    print("🌕 Spacecraft entered Lunar Gravity! Handing over to AI Lander Simulation...")

    eval_env = CustomRewardLagerWrapper(gym.make("LunarLander-v3", render_mode="human"))
    obs, info = eval_env.reset()
    done = False
    pygame.init()
    pygame.display.set_caption("Lunar Lander - Final Descent")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        eval_env.render()
        done = terminated or truncated

    eval_env.close()
