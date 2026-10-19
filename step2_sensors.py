import gymnasium as gym

# 1. Start the simulator window
env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

done = False
frame = 0

print("Frame |   X Pos  | Altitude | Fall Spd |   Tilt   | Legs | Action")
print("-" * 68)

while not done:
    # 2. Unpack the 8 sensor readings from the rocket
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs

    # 3. Simple "if/else" test pilot using the sensors (before we train AI)
    if vy < -0.35:
        action = 2  # Falling too fast -> Fire MAIN bottom engine (2)
        action_name = "MAIN ENGINE"
    elif angle > 0.1:
        action = 3  # Tilted left -> Fire RIGHT thruster (3) to straighten
        action_name = "RIGHT THRUST"
    elif angle < -0.1:
        action = 1  # Tilted right -> Fire LEFT thruster (1) to straighten
        action_name = "LEFT THRUST"
    else:
        action = 0  # Stable -> Do nothing (0) to save fuel
        action_name = "COAST"

    # 4. Step the physics forward by 1 frame
    obs, reward, terminated, truncated, info = env.step(action)
    frame += 1

    # 5. Print live sensor dashboard every 15 frames
    if frame % 15 == 0 or terminated:
        legs_status = f"L:{int(left_leg)} R:{int(right_leg)}"
        print(f"{frame:5d} | {x:+8.2f} | {y:8.2f} | {vy:+8.2f} | {angle:+8.2f} | {legs_status} | {action_name}")

    done = terminated or truncated

print("-" * 68)
if left_leg == 1.0 and right_leg == 1.0 and not env.unwrapped.game_over:
    print("TOUCHDOWN! Both legs are on the ground.")
else:
    print("CRASHED OR OUT OF BOUNDS!")

env.close()
