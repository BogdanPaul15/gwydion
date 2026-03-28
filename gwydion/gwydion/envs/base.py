from typing import List, Optional

import time
from pathlib import Path
from datetime import datetime
from statistics import mean
import yaml

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from gwydion.rewards import RewardStrategy
from gwydion.deployments import build_deployment_list
from gwydion.actions import build_action_set
from .util import save_episode_stats

ACTION_DO_NOTHING = 0

class BaseEnv(gym.Env):
    """Abstract Base Class for Kubernetes Horizontal Auto-scaling Environments.
    
    This class provides a common interface and shared logic for Reinforcement Learning
    environments controlling pod replication in a K8s cluster or simulation.

    Attributes:
        _cfg (dict): The raw configuration dictionary parsed from the YAML file.
        _deployment_cfgs (List[dict]): List of raw deployment configurations 
            extracted from the config file.
        k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
        name (str): The unique name of the environment.
        num_apps (int): The number of managed deployments.
        deployments_names (list[str]): Names of the K8s deployments.
        deployment_list (List[Deployment]): A list of Deployment objects 
            representing the current state and metrics for each active K8s deployment.
        reward_strategy (RewardStrategy): The reward objective function.
        waiting_period (int): Seconds to wait after a scaling action (real K8s only).
        constraint_min_pod_replicas (bool): True if a scaling action attempted to reduce pod
            replicas below the minimum allowed for any deployment.
        constraint_max_pod_replicas (bool): True if a scaling action attempted to increase pod
            replicas above the maximum allowed for any deployment.
        max_steps (int): The maximum number of steps allowed per episode.
        current_step (int): Current step count in the active episode.
        episode_count (int): The total number of episodes completed since initialization.
        terminated (bool): Flag set to True if the agent reaches a terminal state
            (positive or negative).
        episode_over (bool): Flag for reaching max steps (truncated).
        total_reward (float): Accumulated reward for the current episode.
        info (dict): A dictionary containing auxiliary information complementing observation.
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
        _actions (List[Action]): The set of available scaling commands built from config.
        num_actions (int): Total count of possible scaling actions.
        action_space (gym.spaces.MultiDiscrete): A 2-dimensional action vector where the first
            element selects which deployment to scale (0 to num_apps - 1) and the second element
            defines the scaling action to perform (0 to num_actions - 1).
        observation_space (gym.spaces.Box): A multi-dimensional continuous space representing the
            state of the cluster (e.g., current pod counts, traffic)
        file_results (str): A CSV file used to save the episode metrics.
        df (Optional[pd.DataFrame]): The primary dataset containing historical observations metrics
            (e.g., CPU, memory, traffic) used to drive the simulation.
    """
    def __init__(self, config_path: str, reward_strategy: RewardStrategy = None):
        """Initializes the BaseEnv with scaling constraints and core attributes.

        Args:
            config_path (str): 
            reward_strategy (RewardStrategy): The reward objective function.
        """
        super().__init__()

        self._cfg = self._load_config(config_path)
        self._deployments_cfgs = self._cfg["deployments"]
        env_cfg = self._cfg["env"]
        actions_cfg = self._cfg["env"]["actions"]

        self.k8s = env_cfg["k8s"]
        self.name = env_cfg["name"]
        self.num_apps = len(self._deployments_cfgs)
        self.deployments_names = [d["name"] for d in self._deployments_cfgs]
        self.deployment_list = build_deployment_list(self._deployments_cfgs, self.k8s)
        self.reward_strategy = reward_strategy
        self.waiting_period = env_cfg["waiting_period"]
        self.__version__ = env_cfg["version"]

        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        self.max_steps = env_cfg["max_steps"]
        self.current_step = 0
        self.episode_count = 0
        self.terminated = False
        self.episode_over = False
        self.total_reward = 0
        self.info = {}

        self.none_counter = 0
        self.traffic = []

        self.avg_pods = []
        self.avg_latency = []
        self.time_start = 0
        self.execution_time = 0

        self._actions = build_action_set(actions_cfg)
        self.num_actions = len(self._actions)
        self.action_space = spaces.MultiDiscrete([self.num_apps, self.num_actions])
        self.action_stats = [0 for _ in range(self.num_actions)]
        self.observation_space = None

        # TODO: MODIFY THIS
        # self.obs_file = f"{self.name}_observations.csv"
        self.file_results = "results.csv"

        if not self.k8s:
            self._load_dataset()
            self.traffic = self.simulation_traffic(env_cfg["target_deployment"])

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Reads and parses the YAML configuration file from the specified path.

        Args:
            config_path (str): The filesystem path to the YAML configuration file.

        Returns:
            dict: The parsed configuration data as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist at the provided path.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cfg

    def _load_dataset(self):
        """Loads the simulation dataframe using deployment metadata.

        This method enables the environment to simulate cluster behavior using
        real-world observation data when not connected to a live K8s cluster.

        Raises:
            FileNotFoundError: If the observation CSV is missing from the expected
                data directory.
        """
        if not self.k8s:
            # Get namespace from the first deployment in the list
            namespace = self.deployment_list[0].namespace
            base_dir = Path(__file__).resolve().parents[2]
            path = base_dir / "datasets" / "real" / namespace / "v1" / f"{self.name}_observation.csv"
            print(path)

            try:
                self.df = pd.read_csv(path)
                # logging.info(f"[Base] Dataset loaded from {path}")
            except FileNotFoundError:
                print("ERROR")
                # logging.error(f"[Base] Could not find dataset at {path}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalizes the observation vector using the high bounds of the space.
        
        Note:
            TODO: Normalization can be done with a Gymnasium wrapper.
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

        self.current_step = 0
        self.none_counter = 0
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

        self.deployment_list = build_deployment_list(self._deployments_cfgs, self.k8s)

        for deployment in self.deployment_list:
            deployment.initialize_metrics()

        # Note: Child class will implement the actual return.
        # This is a structural placeholder
        return np.array([], dtype=np.float32), self.info

    def render(self, _mode='human', _close=False) -> None:
        """Renders the environment state."""
        return

    def close(self) -> None:
        """Cleans up resources used by the environment."""
        return

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Performs an environment step using the given action.

        Args:
            action (tuple[int, int]):
                A tuple where the first element is the deployment index and the second is the action index.

        Returns:
            tuple: (observation, reward, terminated, episode_over, info)
                observation (np.ndarray): The new observation after the action.
                reward (float): The reward for this step.
                terminated (bool): Whether the episode has terminated.
                episode_over (bool): Whether the episode has reached its maximum steps.
                info (dict): Additional information about the step.
        """
        deployment_id, action_id = action

        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        self.take_action(action_id, deployment_id)

        if self.k8s:
            if action_id != ACTION_DO_NOTHING and not (self.constraint_min_pod_replicas or self.constraint_max_pod_replicas):
                time.sleep(self.waiting_period)

            for d in self.deployment_list:
                d.update_k8s_obs()
        else:
            self.simulation_update()

        reward = self.reward

        self.total_reward += reward
        self.avg_pods.append(sum(d.num_pods for d in self.deployment_list))
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

        if self.current_step == self.max_steps:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_episode_stats(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        return np.array(ob), reward, self.terminated, self.episode_over, self.info

    # TODO: this simulation mode does work only for online_boutique simulation strategy
    # redis environment has a different strategy (the redis strategy is prioritizing the deployment on which the last
    # action was taken)
    # Note: There is a situation when it tries to scale beyond its limits, and the system ends up
    # getting a random sample, instead of looking for a situation with the same number of pods
    def simulation_update(self) -> None:
        """Updates the simulation environment state using historical data.

        This method samples from the loaded dataset to simulate the next environment state
        for each deployment, updating pod counts and metrics. It supports two strategies:
        - On the first step, it samples a random state for all deployments.
        - On subsequent steps, it filters the dataset to match the current and previous pod counts,
          and applies the observed traffic pattern, then samples a matching state.

        Updates the following for each deployment:
            - num_pods
            - num_previous_pods
            - metrics (cpu_usage, mem_usage, received_traffic, transmit_traffic, latency)
            - desired_replicas (via update_desired_replicas)
        """
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

    def take_action(self, deployment_id: int, action: int) -> None:
        """Executes the specified action on the given deployment.

        Increments the step counter, updates episode status, tracks action statistics,
        and invokes the selected action's execute method for the target deployment.

        Args:
            deployment_id (int): The index of the deployment to be scaled.
            action (int): The index of the action in the _actions list.
        """
        self.current_step += 1

        if self.current_step == self.max_steps:
            self.episode_over = True

        self.action_stats[action] += 1
        selected = self._actions[action]

        selected.execute(self, deployment_id)

    @property
    def reward(self):
        """Returns the current reward as computed by the reward strategy."""
        return self.reward_strategy.get_reward(self)

    def get_state(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        raise NotImplementedError
