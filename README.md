# Lunar Lander Reinforcement Learning

A reinforcement-learning project using Gymnasium LunarLander-v3 and PPO (Proximal Policy Optimization).

## Results

- PPO mean reward: 64.08 over 300 episodes.
- Random evaluation mean reward: -181.96 over 300 episodes.

## Project Files

- random_agent.py - Random-action baseline.
- train_ppo.py - Trains the PPO model.
- evaluate_ppo.py - Evaluates PPO.
- evaluate_random.py - Evaluates the random baseline.
- plot_results.py - Generates graphs.
- results/figures/ - Generated plots and summaries.

## Future Work

- Experiment with reward shaping.
- Investigate reward hacking.
- Improve evaluation reproducibility.
