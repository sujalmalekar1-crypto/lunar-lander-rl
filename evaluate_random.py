import csv
from pathlib import Path

import gymnasium as gym

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "results" / "episode_log.csv"
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = []
env = gym.make("LunarLander-v3")

try:
    for episode in range(1, 101):
        observation, info = env.reset(seed=episode)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        rows.append({
            "timestamp": "",
            "agent": "random_eval",
            "episode_steps": steps,
            "total_reward": round(total_reward, 2),
            "terminated": terminated,
            "truncated": truncated,
        })

        print(
            f"Episode {episode:3}/100 | "
            f"Reward: {total_reward:8.2f} | Steps: {steps}"
        )
finally:
    env.close()

write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    if write_header:
        writer.writeheader()
    writer.writerows(rows)

rewards = [row["total_reward"] for row in rows]

print("\n--- Random Evaluation Summary ---")
print(f"Episodes: {len(rewards)}")
print(f"Mean reward: {sum(rewards) / len(rewards):.2f}")
print(f"Best reward: {max(rewards):.2f}")
print(f"Worst reward: {min(rewards):.2f}")
