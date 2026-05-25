import numpy as np
from gymnasium import spaces

from .base import BaseEnv

ID_REDIS_LEADER = 0
ID_REDIS_FOLLOWER = 1

class Redis(BaseEnv):
    """Horizontal Scaling for Redis in K8s - an Gymansium gym environment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = self.get_observation_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        return self.get_state(), self.info

    def get_observation_space(self) -> spaces.Box:
        leader, follower = self.deployment_list[ID_REDIS_LEADER], self.deployment_list[ID_REDIS_FOLLOWER]
        return spaces.Box(
            low=np.array([
                leader.min_pods, # Number of pods -- leader
                leader.min_pods, # Desired number of pods -- leader
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                0, # Traffic In (in KiB)
                0, # Traffic Out (in KiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                follower.min_pods, # Number of pods -- follower
                follower.min_pods, # Desired number of pods -- follower
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                0, # Traffic In (in KiB)
                0, # Traffic Out (in KiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                0, # None counter
            ]),
            high=np.array([
                leader.max_pods, # Number of pods -- leader
                leader.max_pods, # Desired number of pods -- leader
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                20000, # Traffic In (in KiB)
                20000, # Traffic Out (in KiB)
                # 1000, # CPU forecast (in m)
                # 1000, # MEM forecast (in MiB)
                follower.max_pods, # Number of pods -- follower
                follower.max_pods, # Desired number of pods -- leader
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                20000, # Traffic In (in KiB)
                20000, # Traffic Out (in KiB)
                # 1000, # CPU forecast (in m)
                # 1000, # MEM forecast (in MiB)
                25, # None counter
            ]),
            dtype=np.float32
        )

    def get_state(self) -> np.ndarray:
        leader, follower = self.deployment_list[ID_REDIS_LEADER], self.deployment_list[ID_REDIS_FOLLOWER]
        return np.array([
            leader.num_pods,
            leader.desired_replicas,
            leader.metrics["cpu_usage"],
            leader.metrics["mem_usage"],
            leader.metrics["traffic_in"],
            leader.metrics["traffic_out"],
            # leader.cpu_forecast, # CPU forecast (in m)
            # leader.mem_forecast, # MEM forecast (in MiB)
            follower.num_pods,
            follower.desired_replicas,
            follower.metrics["cpu_usage"],
            follower.metrics["mem_usage"],
            follower.metrics["traffic_in"],
            follower.metrics["traffic_out"],
            # follower.cpu_forecast, # CPU forecast (in m)
            # follower.mem_forecast, # MEM forecast (in MiB)
            self.none_counter,
        ], dtype=np.float32)
