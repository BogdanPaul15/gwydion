from statistics import mean
import time
import csv
import datetime

from datetime import datetime

import numpy as np
from gymnasium import spaces

from gwydion.envs import base
from gwydion.envs.deployment import get_redis_deployment_list
from gwydion.envs.util import get_cost_reward, get_latency_reward_redis, save_to_csv, get_num_pods

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

###STEP COUNTER###

MAX_STEPS = 25  # MAX Number of steps per episode

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

# Deployments
DEPLOYMENTS = ["redis-leader", "redis-follower"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_MASTER = 0
ID_SLAVE = 1

# Reward objectives
LATENCY = 'latency'
COST = 'cost'

class Redis(base.BaseEnv):
    """Horizontal Scaling for Redis in K8s - an Gymansium gym environment."""
    def __init__(self, k8s=False, goal_reward=base.COST, waiting_period=5):
        super().__init__(
            name="redis_gym",
            num_apps=2,
            deployments=["redis-leader", "redis-follower"],
            k8s=k8s,
            goal_reward=goal_reward,
            waiting_period=waiting_period
        )

        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        self.observation_space = self.get_observation_space()

        # TODO remove this
        self.file_results = "results.csv"

        if not k8s:
            self.load_dataset()
            self.traffic = self.simulation_traffic("redis-leader")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return self.get_state(), self.info
    
    def take_action(self, action, id):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.none_counter = 0
            self.episode_over = True

        self.action_stats[action] += 1

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            self.none_counter += 1
            print("[Take Action] SELECTED ACTION: DO NOTHING ...")
            pass

        elif action == ACTION_ADD_1_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deploymentList[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deploymentList[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(7, self)

        else:
            print('[Take Action] Unrecognized Action: ' + str(action))

    def calculate_reward(self):
        reward = 0
        if self.goal_reward == base.COST:
            reward = get_cost_reward(self.deploymentList)
            if reward !=2 and self.none_counter > 2:
                reward = -self.none_counter
        elif self.goal_reward == base.LATENCY:
            reward = get_latency_reward_redis(ID_MASTER, self.deploymentList)
            if self.none_counter > 2:
                reward = -self.none_counter * 250

        return reward

    def get_observation_space(self):
        return spaces.Box(
            low=np.array([
                self.min_pods, # Number of pods -- leader
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                0, # CPU forecast (in m)
                0, # MEM forecast (in MiB)
                self.min_pods, # Number of pods -- follower
                0, # CPU Usage (in m)
                0, # MEM Usage (in MiB)
                0, # CPU forecast (in m)
                0, # MEM forecast (in MiB)
            ]),
            high=np.array([
                self.max_pods, # Number of pods -- leader
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                1000, # CPU forecast (in m)
                1000, # MEM forecast (in MiB)
                self.max_pods, # Number of pods -- follower
                1000, # CPU Usage (in m)
                1000, # MEM Usage (in MiB)
                1000, # CPU forecast (in m)
                1000, # MEM forecast (in MiB)
            ]),
            dtype=np.float32
        )

    def get_state(self):
        ob = (
            self.deploymentList[0].num_pods, # Number of pods -- leader
            self.deploymentList[0].cpu_usage, #  CPU Usage (in m)
            self.deploymentList[0].mem_usage, # MEM Usage (in MiB)
            self.deploymentList[0].cpu_forecast, # CPU forecast (in m)
            self.deploymentList[0].mem_forecast, # MEM forecast (in MiB)
            self.deploymentList[0].num_pods, # Number of pods -- follower
            self.deploymentList[0].cpu_usage, #  CPU Usage (in m)
            self.deploymentList[0].mem_usage, # MEM Usage (in MiB)
            self.deploymentList[0].cpu_forecast, # CPU forecast (in m)
            self.deploymentList[0].mem_forecast, # MEM forecast (in MiB)
        )

        # return self.normalize(ob)
        return ob
