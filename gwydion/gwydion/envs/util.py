from typing import Callable, Type, Tuple, Any
import csv
import os
import time
import functools


def save_episode_stats(path: str, episode: int, avg_pods: float, avg_latency: float, reward: float, execution_time: float) -> None:
    """TODO"""
    file_exists = os.path.isfile(path)

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

###REWARD FUNCTIONS###
# def get_cost_reward(deployment_list):
#     reward = 0

#     for d in deployment_list:
#         num_pods = d.num_pods
#         desired_replicas = d.desired_replicas
#         if num_pods == desired_replicas:
#             reward += 1

#     # if reward == 2:
#         # return reward
#     # else:
#         # return 0

#     return reward


# def get_latency_reward_redis(ID_MASTER, deployment_list):
#     # Calculate the redis latency based on the redis exporter
#     reward = float(deployment_list[ID_MASTER].latency)
#     if reward > 250.0:
#         reward = -250  # highest penalty over 250 ms
#     else:
#         reward = -float(deployment_list[ID_MASTER].latency)  # negative reward

#     return reward


# def get_latency_reward_online_boutique(ID_recommendation, deployment_list):
#     # Calculate the latency based on the GET / POST requests
#     reward = float(deployment_list[ID_recommendation].latency)
#     if reward > 3000.0:
#         reward = -3000  # highest penalty over 3 s
#     else:
#         reward = -float(deployment_list[ID_recommendation].latency)  # negative reward

#     return reward


# def get_num_pods(deployment_list):
#     n = 0
#     for d in deployment_list:
#         n += d.num_pods

#     return n
