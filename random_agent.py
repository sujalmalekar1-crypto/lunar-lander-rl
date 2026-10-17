import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
CSV_FILE = RESULTS_DIR / "episode_log.csv"
RESULTS_DIR.mkdir(exist_ok=True)

env = gym.make("LunarLander-v3", render_mode="human")

try:
    observation, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < 1000:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        time.sleep(0.03)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent": "random",
        "episode_steps": steps,
        "total_reward": round(total_reward, 2),
        "terminated": terminated,
        "truncated": truncated,
    }

    with CSV_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"Episode steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Saved results to: {CSV_FILE}")

finally:
    env.close()
