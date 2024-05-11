from stable_baselines3 import PPO
from gymnasium import Env
from gymnasium.wrappers import RecordVideo


def learn(iters: int, env: Env, save_path: str, record_path: str | None = None):
    model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=0.3)
    try:
        model.learn(iters)
    finally:
        model.save(save_path)

    if record_path is not None:
        env = RecordVideo(env, record_path, episode_trigger=lambda e: e % 1 == 0)

        obs, info = env.reset()
        done = False
        env.start_video_recorder()
        steps = 0
        while not done and steps <= 30000:
            action = model.predict(obs)[0]
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        env.close_video_recorder()
        env.close()
