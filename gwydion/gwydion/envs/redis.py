import csv

import numpy as np
from gymnasium import spaces

from gwydion.envs import base

class Redis(base.BaseEnv):
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
                self.deployment_list[0].min_pods, # Number of pods -- leader
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                self.deployment_list[1].min_pods, # Number of pods -- follower
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                # 0, # CPU forecast (in m)
                # 0, # MEM forecast (in MiB)
                0, # None counter
            ]),
            high=np.array([
                self.deployment_list[0].max_pods, # Number of pods -- leader
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                # 1000, # CPU forecast (in m)
                # 1000, # MEM forecast (in MiB)
                self.deployment_list[1].max_pods, # Number of pods -- follower
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
            self.deployment_list[0].num_pods, # Number of pods -- leader
            self.deployment_list[0].metrics["cpu_usage"], #  CPU Usage (in m)
            self.deployment_list[0].metrics["mem_usage"], # MEM Usage (in MiB)
            # self.deployment_list[0].cpu_forecast, # CPU forecast (in m)
            # self.deployment_list[0].mem_forecast, # MEM forecast (in MiB)
            self.deployment_list[1].num_pods, # Number of pods -- follower
            self.deployment_list[1].metrics["cpu_usage"], #  CPU Usage (in m)
            self.deployment_list[1].metrics["mem_usage"], # MEM Usage (in MiB)
            # self.deployment_list[1].cpu_forecast, # CPU forecast (in m)
            # self.deployment_list[1].mem_forecast, # MEM forecast (in MiB)
        )

        # return self.normalize(ob)
        return ob

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file = open(obs_file, 'a+', encoding='utf-8', newline='')
        fields = []
        with file:
            fields.append('date')
            for d in self.deployment_list:
                fields.append(d.name + '_num_pods')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_mem_usage')
                fields.append(d.name + '_latency')

            writer = csv.DictWriter(file, fieldnames=fields)
            # TODO this writes an independent header for each row
            # writer.writeheader()
            writer.writerow(
                {'date': date,
                 'redis-leader_num_pods': int(f"{obs[0]}"),
                 'redis-leader_cpu_usage': int(f"{obs[1]}"),
                 'redis-leader_mem_usage': int(f"{obs[2]}"),
                 'redis-leader_latency': float(f"{latency:.3f}"),
                 'redis-follower_num_pods': int(f"{obs[3]}"),
                 'redis-follower_cpu_usage': int(f"{obs[4]}"),
                 'redis-follower_mem_usage': int(f"{obs[5]}"),
                 }
            )
