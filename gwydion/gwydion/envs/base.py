from typing import List, Optional

from datetime import datetime
from statistics import mean
import time

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from gwydion.envs.util import get_num_pods, save_to_csv
from gwydion.envs.rewards import RewardStrategy

MIN_REPLICATION = 1
MAX_REPLICATION = 8
MAX_STEPS = 25
ACTION_MOVES_COUNT = 15

ID_DEPLOYMENTS = 0
ID_MOVES = 1

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


class BaseEnv(gym.Env):
    """Abstract Base Class for Kubernetes Horizontal Scaling Environments.
    
    This class provides a common interface and shared logic for Reinforcement Learning
    environments controlling pod replication in a K8s cluster or simulation.

    Attributes:
        name (str): The unique name of the environment.
        num_apps (int): The number of managed deployments.
        deployments_name (list[str]): Name of the K8s deployments.
        k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
        reward_strategy (RewardStrategy): The reward objective function.
        waiting_period (int): Seconds to wait after a scaling action (real K8s only).
        min_pods (int): Minimum replica count allowed per deployment.
        max_pods (int): Maximum replica count allowed per deployment.
        constraint_min_pod_replicas (bool): Flag set to True if a scaling action 
            attempted to drop below MIN_REPLICATION.
        constraint_max_pod_replicas (bool): Flag set to True if a scaling action
            attempted to exceed MAX_REPLICATION.
        current_step (int): Current step count in the active episode.
        episode_count (int): The total number of episodes completed since initialization.
        terminated (bool): Flag set to True if the agent reaches a terminal state
            (positive or negative).
        episode_over (bool): Flag for reaching max steps (truncated).
        total_reward (float): Accumulated reward for the current episode.
        info (dict): A dictionary containing auxiliary information complementing observation.
        num_actions (int): Total count of possible scaling actions.
        none_counter (int): Count of "Do Nothing" consecutive actions in the current episode.
        action_stats (List[int]): List containing counters for each possible action taken in the
            current episode. The index corresponds to the action ID.
        traffic (List[float]): List of unique traffic values preserved in their original
            appearance order. (Values are collected from a specific deployment)
        avg_pods (List[int]): List containing the number of pods for each deployment tracked in
            the current episode (e.g., index 0 corresponds to the first deployment).
        avg_latency (List[float]): List containing the latency values for Deployment 0, recorded
            at each step of the current episode (e.g., index 0 corresponds to the first step).
        time_start (float): The timestamp (in seconds) representing when the episode started.
        execution_time (float): Total duration (in seconds) taken to complete the current episode.
        deployment_list (List[BaseDeploymentWorkload]): A list of BaseDeploymentWorkload objects representing
            the current state and metrics for each active K8s deployment.
        action_space (gym.spaces.MultiDiscrete): A 2-dimensional action vector where the first
            element selects which deployment to scale (0 to num_apps - 1) and the second element
            defines the scaling action to perform (0 to num_actions - 1).
        observation_space (gym.spaces.Box): A multi-dimensional continuous space representing the
            state of the cluster (e.g., current pod counts, traffic)
        df (Optional[pd.DataFrame]): The primary dataset containing historical observations metrics
            (e.g., CPU, memory, traffic) used to drive the simulation.
    """
    def __init__(self, name: str, num_apps: int, deployments: List[str],
                 k8s: bool = False, reward_strategy: RewardStrategy = None, waiting_period: int = 5):
        """Initializes the BaseEnv with scaling constraints and core attributes.

        Args:
            name (str): The unique name of the environment.
            num_apps (int): The number of managed deployments.
            k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
            reward_strategy (RewardStrategy): The reward objective function.
            waiting_period (int): Seconds to wait after a scaling (real K8s only).

        """
        super(BaseEnv, self).__init__()

        self.name = name
        self.num_apps = num_apps
        self.deployments_names = deployments
        self.k8s = k8s
        self.reward_strategy = reward_strategy
        self.waiting_period = waiting_period
        self.__version__ = "0.0.1"

        self.min_pods = 1
        self.max_pods = 8

        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        self.current_step = 0
        self.episode_count = 0
        self.terminated = False
        self.episode_over = False
        self.total_reward = 0
        self.info = {}

        # TODO: replace 
        self.num_actions = ACTION_MOVES_COUNT
        self.none_counter = 0
        self.action_stats = [0 for _ in range(self.num_actions)]
        self.traffic = []

        self.avg_pods = []
        self.avg_latency = []
        self.time_start = 0
        self.execution_time = 0

        self.deployment_list = []
        self.action_space = spaces.MultiDiscrete([num_apps, self.num_actions])
        self.observation_space = None

        # TODO: MODIFY THIS
        # self.obs_file = f"{self.name}_observations.csv"

        self.df = None

    def load_dataset(self):
        """Loads the simulation dataframe using deployment metadata.
        
        This must be called AFTER self.deployment_list is initialized in the child class.
        """
        if not self.k8s:
            # Get namespace from the first deployment in the list
            namespace = self.deployment_list[0].namespace
            path = f"datasets/real/{namespace}/v1/{self.name}_observation.csv"

            try:
                self.df = pd.read_csv(path)
                # logging.info(f"[Base] Dataset loaded from {path}")
            except FileNotFoundError:
                # logging.error(f"[Base] Could not find dataset at {path}")
                print("ERROR")

    def normalize(self, obs):
        """Normalizes the observation vector using the high bounds of the space.
        
        Note:
            TODO: Normalization can be added with a Gymnasium wrapper.
            Reference: https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.NormalizeObservation

        Args:
            obs: The raw observation.

        Returns:
            np.ndarray
        """
        return obs / self.observation_space.high

    def simulation_traffic(self, deployment: str) -> List[float]:
        """Extracts unique traffic values for the specified deployment.
        
        Args:
            deployment (str): The name of the deployment to extract traffic for
                (e.g., "redis-leader", "frontend")

        Returns:
            List[float]: A list of unique traffic values preserved in  their
                original appearance order.
        """
        column = f"{deployment}_traffic_in"
        if self.df is not None and column in self.df.columns:
            seen = set()
            self.traffic = [
                traffic_value for traffic_value in self.df[column]
                if not (traffic_value in seen or seen.add(traffic_value))
            ]
        return self.traffic

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Resets the environment to an initial state and returns an initial observation.

        This method initializes the random number generator using the provided seed,
        resets all episode-specific counters, flags, and performance metrics, and
        prepares the environment for a new episode.
        
        Args:
            seed (Optional[int]): The seed used to initialize the environment's PRNG.
                Defaults to None.
            options (Optional[dict]): Additional information to specify how to reset the
                environment. Defaults to None.
        
        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): Observation of the initial state.
                - info (dict): A dictionary containing auxiliary information.
        """
        super().reset(seed=seed)

        # TODO: self.none_counter should be added here as well
        self.current_step = 0
        self.total_reward = 0

        self.terminated = False
        self.episode_over = False
        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        self.avg_pods = []
        self.avg_latency = []

        self.time_start = 0
        self.execution_time = 0
        self.info = {}
        self.action_stats = [0 for _ in range(self.num_actions)]

        # Note: self.deployment_list should be reinitialized in the child
        # after calling super().reset()

        # Note: Child class will implement the actual return.
        # This is a structural placeholder
        return np.array([], dtype=np.float32), self.info

    def render(self, mode='human', close=False):
        """Renders the environment state."""
        return

    def close(self) -> None:
        """Cleans up resources used by the environment."""
        return

    def step(self, action):
        app_id, move_id = action

        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        self.take_action(move_id, app_id)

        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING and not (self.constraint_min_pod_replicas or self.constraint_max_pod_replicas):
                time.sleep(self.waiting_period)

            for d in self.deployment_list:
                d.update_k8s_obs()
        else:
            self.simulation_update()

        reward = self.get_reward

        self.total_reward += reward
        self.avg_pods.append(get_num_pods(self.deployment_list))
        # TODO replace 0 with target_id (the target deployment)
        self.avg_latency.append(self.deployment_list[0].metrics["latency"])

        self.info = {
            "reward": f"{self.total_reward:.2f}",
            'avg_pods': f"{mean(self.avg_pods):.3f}",
            'avg_latency': f"{mean(self.avg_latency):.3f}",
            'executionTime': f"{self.execution_time:.3f}"
        }

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TODO: should be called before normalizing the observations
        # TODO: replace 0 with target_id (the target deployment)
        self.save_obs_to_csv(f"{self.name}_observation.csv", np.array(ob), date, self.deployment_list[0].metrics["latency"])

        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        return np.array(ob), reward, self.terminated, self.episode_over, self.info

    # TODO: this simulation mode does work only for online_boutique simulation strategy
    # redis environment has a different strategy (the redis strategy is prioritizing the deployment on which the last
    # action was taken)
    # Note: There is a situation when it tries to scale beyond its limits, and the system ends up
    # getting a random sample, instead of looking for a situation with the same number of pods
    def simulation_update(self):
        if self.current_step == 1:
            sample = self.df.sample()

            for i, name in enumerate(self.deployments_names):
                self.deployment_list[i].num_pods = int(sample[f"{name}_num_pods"].values[0])
                self.deployment_list[i].num_previous_pods = int(sample[f"{name}_num_pods"].values[0])

        else:
            pods = []
            previous_pods = []
            diff = []
            data = self.df

            for i, name in enumerate(self.deployments_names):
                pods.append(self.deployment_list[i].num_pods)
                previous_pods.append(self.deployment_list[i].num_previous_pods)
                aux = pods[i] - previous_pods[i]
                diff.append(aux)
                self.df[f"diff-{name}"] = self.df[f"{name}_num_pods"].diff()

            for i in range(self.num_apps):
                data = data.loc[self.df[f"{self.deployments_names[i]}_num_pods"] == pods[i]]
                data = data.loc[data[f"diff-{self.deployments_names[i]}"] == diff[i]]

                new_traffic = self.traffic.pop(0)

                if data.size == 0:
                    data = data.loc[self.df[f"{self.deployments_names[i]}_num_pods"] == pods[i]]

                self.traffic.append(new_traffic)

                if data.size == 0:
                    data = self.df.loc[self.df[f"{self.deployments_names[i]}_num_pods"] == pods[i]]

            sample = data.sample()

        for i, name in enumerate(self.deployments_names):
            self.deployment_list[i].metrics["cpu_usage"] = int(sample[f"{name}_cpu_usage"].values[0])
            self.deployment_list[i].metrics["mem_usage"] = int(sample[f"{name}_mem_usage"].values[0])
            self.deployment_list[i].metrics["received_traffic"] = int(sample[f"{name}_traffic_in"].values[0])
            self.deployment_list[i].metrics["transmit_traffic"] = int(sample[f"{name}_traffic_out"].values[0])
            self.deployment_list[i].metrics["latency"] = float(f"{sample[f'{name}_latency'].values[0]:.3f}")

        for d in self.deployment_list:
            d.update_desired_replicas()

        return

    def take_action(self, action, id):
        raise NotImplementedError

    @property
    def get_reward(self):
        return self.reward_strategy.get_reward(self)

    def get_state(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        raise NotImplementedError
