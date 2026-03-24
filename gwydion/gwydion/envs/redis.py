import csv

import numpy as np
from gymnasium import spaces

from gwydion.envs import base

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

ID_MASTER = 0

class Redis(base.BaseEnv):
    """Horizontal Scaling for Redis in K8s - an Gymansium gym environment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = self.get_observation_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        return self.get_state(), self.info

    def take_action(self, action, id):
        self.current_step += 1

        # Stop if self.max_steps
        if self.current_step == self.max_steps:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.none_counter = 0
            self.episode_over = True

        self.action_stats[action] += 1

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            self.none_counter += 1
            print("[Take Action] SELECTED ACTION: DO NOTHING ...")

        elif action == ACTION_ADD_1_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deployment_list[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deployment_list[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(7, self)

        else:
            print('[Take Action] Unrecognized Action: ' + str(action))

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
