from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "results" / "episode_log.csv"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

if df.empty:
    raise ValueError("No episode data found.")

summary = df.groupby("agent")["total_reward"].agg(
    episodes="count",
    mean_reward="mean",
    std_reward="std",
    best_reward="max",
    worst_reward="min",
)

summary.to_csv(FIGURES_DIR / "agent_comparison.csv")

plt.figure(figsize=(9, 5))
summary["mean_reward"].plot(kind="bar", rot=0)
plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Mean episode reward")
plt.title("Random Agent vs PPO")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "agent_comparison.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 5))
for agent, group in df.groupby("agent"):
    rewards = group["total_reward"].reset_index(drop=True)
    rolling = rewards.rolling(
        window=min(10, len(rewards)), min_periods=1
    ).mean()
    plt.plot(range(1, len(rewards) + 1), rolling, label=agent)

plt.xlabel("Episode")
plt.ylabel("Rolling mean reward")
plt.title("Episode Rewards by Agent")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "reward_comparison.png", dpi=200)
plt.close()

print(summary.round(2).to_string())
print(f"\nGraphs saved to: {FIGURES_DIR}")
