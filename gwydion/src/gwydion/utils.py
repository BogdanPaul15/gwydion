from typing import Callable, Type, Tuple, Any
import csv
from pathlib import Path
import time
import functools

def save_episode_stats(path: str, episode: int, avg_pods: float, avg_latency: float, reward: float, execution_time: float) -> None:
    """TODO"""
    file_exists = Path(path).exists()

    with open(path, "a+", encoding="utf-8",newline="") as f:
        fields = ["episode", "avg_pods", "avg_latency", "reward", "execution_time"]
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {'episode': episode,
             'avg_pods': float(f"{avg_pods:.2f}"),
             'avg_latency': float(f"{avg_latency:.4f}"),
             'reward': float(f"{reward:.2f}"),
             'execution_time': float(f"{execution_time:.2f}")}
        )

def backoff(
    delay: float = 0.5,
    retries: int = 3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on specified exceptions.

    Args:
        delay (float): Initial delay in seconds before retrying. Default is 2.
        retries (int): Maximum number of attempts. Default is 3.
        exceptions (tuple[type[BaseException], ...]): Exception types to catch and retry on. 
            Default is (Exception,).

    Returns:
        Callable: The decorated function with retry logic.

    Usage:
        @backoff(delay=2, retries=3, exceptions=(SomeException,))
        def my_func(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_retry = 0
            current_delay = delay
            while current_retry < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    current_retry += 1
                    if current_retry >= retries:
                        raise
                    print(f"Failed to execute function '{func.__name__}' due to: {e}. \
                          Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2
        return wrapper
    return decorator

import pandas as pd
from matplotlib import pyplot as plt


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    obs = env.reset()

    print("------------Testing -----------------")

    for e in range(n_episodes):
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                episode_rewards.append(reward_sum)
                print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
                reward_sum = 0
                obs = env.reset()
                break

    env.close()

    # Free memory
    del model, env

    # Plot the episode reward over time
    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(fig_name, dpi=250, bbox_inches='tight')
