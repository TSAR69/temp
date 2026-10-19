import gymnasium as gym
import numpy as np

class CustomRewardLander(gym.Wrapper):
    """
    Custom Reward Shaping Wrapper for LunarLander-v3.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_potential = None

    def calculate_potential(self, obs):
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs

        # 1. Distance score (closer to pad 0,0 is better)
        distance_score = -100.0 * np.sqrt(x**2 + y**2)

        # 2. Speed score (slower descent is better)
        speed_score = -100.0 * np.sqrt(vx**2 + vy**2)

        # 3. Upright orientation score
        angle_score = -100.0 * abs(angle) - 20.0 * abs(angular_vel)

        # 4. Leg contact bonus
        leg_bonus = 10.0 * left_leg + 10.0 * right_leg

        return distance_score + speed_score + angle_score + leg_bonus

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self.calculate_potential(obs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs

        # A. Potential shaping (progress toward goal)
        current_potential = self.calculate_potential(obs)
        custom_reward = current_potential - self.prev_potential
        self.prev_potential = current_potential

        # B. Fuel penalty based on action
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

        return obs, custom_reward, terminated, truncated, info

# 1. Wrap our environment
env = CustomRewardLander(gym.make("LunarLander-v3", render_mode="human"))
obs, info = env.reset()

done = False
total_custom_score = 0.0

print("Running simulation with your custom reward function...")

while not done:
    # Simple autopilot test policy
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs
    if vy < -0.35:
        action = 2
    elif angle > 0.1:
        action = 3
    elif angle < -0.1:
        action = 1
    else:
        action = 0

    obs, reward, terminated, truncated, info = env.step(action)
    total_custom_score += reward
    done = terminated or truncated

print(f"Episode finished! Total Custom Reward Score: {total_custom_score:.2f}")
env.close()
