
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Create parallel training environments
env = make_vec_env("LunarLander-v3", n_envs=4)

# PPO training configuration
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    seed=42,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000 // env.num_envs,
    save_path=str(MODEL_DIR),
    name_prefix="ppo_lunar_lander",
)

try:
    model.learn(
        total_timesteps=200_000,
        callback=checkpoint_callback,
        progress_bar=False,
    )
    model.save(MODEL_DIR / "ppo_lunar_lander_final")
    print(f"\nTraining complete. Model saved to: {MODEL_DIR}")
finally:
    env.close()
