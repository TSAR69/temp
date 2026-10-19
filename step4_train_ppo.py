import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class CustomRewardLander(gym.Wrapper):
    """
    Custom Reward Shaping Wrapper for LunarLander-v3.
    """
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

        # A. Potential shaping
        current_potential = self.calculate_potential(obs)
        custom_reward = current_potential - self.prev_potential
        self.prev_potential = current_potential

        # B. Fuel penalty
        if action == 2:  # Main engine
            custom_reward -= 0.30
        elif action in [1, 3]:  # Side thrusters
            custom_reward -= 0.03

        # C. Time penalty
        custom_reward -= 0.05

        # D. Terminal landing / crash rewards
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

if __name__ == "__main__":
    print("Opening live training environment (window will appear)...")
    # render_mode="human" forces the simulator window to open live during training
    train_env = CustomRewardLander(gym.make("LunarLander-v3", render_mode="human"))

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

    print("Training PPO rocket pilot live on screen (this will take longer)...")
    # Reduced steps slightly so you don't have to wait hours, but enough to see improvement
    model.learn(total_timesteps=100_000)
    model.save("ppo_rocket_pilot_live")
    print("Training complete! Model saved.")

    train_env.close()
