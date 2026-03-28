import csv
import os

import numpy as np
from gymnasium import spaces

from gwydion.envs import BaseEnv

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

    def get_observation_space(self):
        return spaces.Box(
            low=np.array([
                self.deployment_list[ID_REDIS_LEADER].min_pods, # Number of pods -- leader
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                self.deployment_list[ID_REDIS_FOLLOWER].min_pods, # Number of pods -- follower
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                0, # None counter
            ]),
            high=np.array([
                self.deployment_list[ID_REDIS_LEADER].max_pods, # Number of pods -- leader
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                # 1000, # CPU forecast (in m)
                # 1000, # MEM forecast (in MiB)
                self.deployment_list[ID_REDIS_FOLLOWER].max_pods, # Number of pods -- follower
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                # 1000, # CPU forecast (in m)
                # 1000, # MEM forecast (in MiB)
                10, # None counter
            ]),
            dtype=np.float32
        )

    def get_state(self):
        ob = (
            self.deployment_list[ID_REDIS_LEADER].num_pods, # Number of pods -- leader
            self.deployment_list[ID_REDIS_LEADER].metrics["cpu_usage"], #  CPU Usage (in m)
            self.deployment_list[ID_REDIS_LEADER].metrics["mem_usage"], # MEM Usage (in MiB)
            # self.deployment_list[0].cpu_forecast, # CPU forecast (in m)
            # self.deployment_list[0].mem_forecast, # MEM forecast (in MiB)
            self.deployment_list[ID_REDIS_FOLLOWER].num_pods, # Number of pods -- follower
            self.deployment_list[ID_REDIS_FOLLOWER].metrics["cpu_usage"], #  CPU Usage (in m)
            self.deployment_list[ID_REDIS_FOLLOWER].metrics["mem_usage"], # MEM Usage (in MiB)
            # self.deployment_list[1].cpu_forecast, # CPU forecast (in m)
            # self.deployment_list[1].mem_forecast, # MEM forecast (in MiB)
        )

        # return self.normalize(ob)
        return ob

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file_exists = os.path.isfile(obs_file)

        with open(obs_file, "a+", encoding="utf-8", newline="") as f:
            fields = ["date"]
            for d in self.deployment_list:
                fields.extend([
                    f"{d.name}_num_pods",
                    f"{d.name}_cpu_usage",
                    f"{d.name}_mem_usage",
                ])
            fields.append("redis-leader_latency")

            writer = csv.DictWriter(f, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            row_data = {
                "date": date,
                "redis-leader_latency": float(f"{latency:.3f}")
            }

            for i, d in enumerate(self.deployment_list):
                idx = i * 3
                row_data[f"{d.name}_num_pods"] = int(f"{obs[idx]}")
                row_data[f"{d.name}_cpu_usage"] = int(f"{obs[idx + 1]}")
                row_data[f"{d.name}_mem_usage"] = int(f"{obs[idx + 2]}")

            writer.writerow(row_data)
