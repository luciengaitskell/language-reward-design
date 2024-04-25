import gymnasium as gym

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="human")
observation, info = env.reset(seed=42)

if False:
    for _ in range(1000):

        action = 1  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

_ = input("wait (press enter to exit)")
env.close()
