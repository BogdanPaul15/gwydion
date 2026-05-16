from typing import Optional

import time
from pathlib import Path
from datetime import datetime
from statistics import mean
import logging
import yaml
import urllib3

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from gwydion.rewards import RewardStrategy
from gwydion.deployments import build_deployment_list
from gwydion.actions import build_action_set
from gwydion.actions import build_action_space
from gwydion.simulation import build_simulation_strategies
from gwydion.utils import save_episode_stats

logger = logging.getLogger(__name__)
logging.disable(logging.ERROR)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BaseEnv(gym.Env):
    """Abstract Base Class for Kubernetes Horizontal Auto-scaling Environments.
    
    This class provides a common interface and shared logic for Reinforcement Learning
    environments controlling pod replication in a K8s cluster or simulation.

    Attributes:
        seed (Optional[int]): The random seed used to ensure reproducibility across episodes.
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
        _action_adapter (ActionSpaceAdapter): Adapter handling the conversion of RL continuous/discrete actions into discrete environment instructions.
        action_space (gym.spaces.MultiDiscrete): A 2-dimensional action vector where the first
            element selects which deployment to scale (0 to num_apps - 1) and the second element
            defines the scaling action to perform (0 to num_actions - 1).
        observation_space (gym.spaces.Box): A multi-dimensional continuous space representing the
            state of the cluster (e.g., current pod counts, traffic)
        file_results (str): A CSV file used to save the episode metrics.
        episode_buffer (list): A temporary buffer storing metrics or transitions for the current episode before writing to disk.
        df (Optional[pd.DataFrame]): The primary dataset containing historical observations metrics
            (e.g., CPU, memory, traffic) used to drive the simulation.
    """
    def __init__(self, config_path: str, reward_strategy: RewardStrategy, seed: Optional[int] = None):
        """Initializes the BaseEnv with scaling constraints and core attributes.

        Args:
            config_path (str): The file path to the YAML configuration file.
            reward_strategy (RewardStrategy): The reward objective function.
            seed (Optional[int]): The random seed for reproducibility. Defaults to None.
        """
        super().__init__()
        self.seed = seed

        self._cfg = self._load_config(config_path)
        self._deployments_cfgs = self._cfg["deployments"]
        self._env_cfg = self._cfg["env"]
        actions_cfg = self._cfg["env"]["actions"]

        self.k8s = self._env_cfg["k8s"]
        self.name = self._env_cfg["name"]
        self.num_apps = len(self._deployments_cfgs)
        self.deployments_names = [d["name"] for d in self._deployments_cfgs]
        self.deployment_list = build_deployment_list(self._deployments_cfgs, self.k8s, 
                                                     self._env_cfg["host"], self._env_cfg["token"],
                                                     self._env_cfg["prometheus_url"])
        self.reward_strategy = reward_strategy
        self.waiting_period = self._env_cfg["waiting_period"]
        self.__version__ = self._env_cfg["version"]

        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        self.max_steps = self._env_cfg["max_steps"]
        self.current_step = 0
        self.episode_count = 0
        self.terminated = False
        self.episode_over = False
        self.total_reward = 0
        self.info = {}

        self.none_counter = 0

        self.avg_pods = []
        self.avg_latency = []
        self.time_start = 0
        self.execution_time = 0

        self._actions = build_action_set(actions_cfg)
        self.num_actions = len(self._actions)
        space_type = self._env_cfg["action_space_type"]
        self._action_adapter = build_action_space(space_type, self.num_apps, self.num_actions)
        self.action_space = self._action_adapter.gym_space
        self.action_stats = [0 for _ in range(self.num_actions)]
        self.observation_space: spaces.Box = None # type: ignore

        # TODO: modify the path where obs and results file are saved
        self.obs_file = f"{self.name}_observation.csv"
        self.file_results = "results.csv"

        self.episode_buffer = []

        if not self.k8s:
            self._load_dataset()

            strategy_cfg = self._env_cfg.get("simulation_strategy", {"type": "default"})
            self.simulation_strategy = build_simulation_strategies(strategy_cfg, seed, df=self.df, deployment_names=self.deployments_names)

        logger.info("Environment: %s | Mode: %s | Strategy: %s | Steps per episode: %d",
            self.name, "K8s" if self.k8s else "Simulation",
            self.reward_strategy.__class__.__name__, self.max_steps)

        for d in self.deployment_list:
            logger.info("  Deployment: %s | Namespace: %s | Pods: [%d, %d]",
                        d.name, d.namespace, d.min_pods, d.max_pods)

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
            base_dir = Path(__file__).resolve().parents[3]
            version = self._env_cfg.get("dataset_version", "v2")
            path = base_dir / "datasets" / "real" / namespace / version / f"{self.name}_observation.csv"
            logger.debug("Loading dataset from %s", path)

            try:
                self.df = pd.read_csv(path)
                for name in self.deployments_names:
                    self.df[f"diff-{name}"] = self.df[f"{name}_num_pods"].diff().fillna(0).astype(int)
                logger.info("Dataset loaded: %s | Rows: %d", path.name, len(self.df))
            except FileNotFoundError as e:
                logger.error("Dataset not found at %s: %s", path, e, exc_info=True)
                raise

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

        self.episode_buffer = []

        self.deployment_list = build_deployment_list(self._deployments_cfgs, self.k8s, 
                                                     self._env_cfg["host"], self._env_cfg["token"],
                                                     self._env_cfg["prometheus_url"])

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
        action_pairs = self._action_adapter.decode(action)

        self.take_action(action_pairs)

        if self.k8s:
            any_scaling = any(not self._actions[a].is_noop for _, a in action_pairs)
            if any_scaling and not (self.constraint_min_pod_replicas or self.constraint_max_pod_replicas):
                time.sleep(self.waiting_period)

            for d in self.deployment_list:
                d.update_obs_k8s()
        else:
            self.simulation_strategy.update(self, action)

        reward = self.reward

        self.total_reward += reward
        self.avg_pods.append(sum(d.num_pods for d in self.deployment_list))
        self.avg_latency.append(self.deployment_list[self._cfg["env"]["target_id"]].metrics["latency"])

        logger.debug("[Step: %d] | Reward: %-6.2f | Total: %.2f | Avg Pods: %.2f",
             self.current_step, reward, self.total_reward,
             mean(self.avg_pods) if self.avg_pods else 0.0)

        self.info = {
            "reward": f"{self.total_reward:.2f}",
            'avg_pods': f"{mean(self.avg_pods):.3f}",
            'avg_latency': f"{np.mean(self.avg_latency):.3f}",
            'executionTime': f"{self.execution_time:.3f}"
        }

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_latency = self.deployment_list[self._cfg["env"]["target_id"]].metrics["latency"]
        # TODO: should be called before normalizing the observations
        self.collect_obs(np.array(ob), date, target_latency)

        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        if self.current_step == self.max_steps:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            self.save_obs_to_csv()
            save_episode_stats(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)
            logger.info("="*100)
            logger.info("EPISODE END: %d | Steps: %d | Reward: %.2f | Avg Pods: %.2f | Avg Latency: %.3f | Time: %.2fs",
                        self.episode_count, self.current_step, self.total_reward,
                        mean(self.avg_pods) if self.avg_pods else 0.0,
                        np.mean(self.avg_latency) if self.avg_latency else 0.0,
                        self.execution_time)
            logger.info("="*100)

        return np.array(ob), reward, self.terminated, self.episode_over, self.info

    def take_action(self, action_pairs: list[tuple[int, int]]) -> None:
        """Executes one scaling action per targeted deployment for this step.

        Increments the step counter, updates episode status, tracks action
        statistics, and invokes each selected action's execute method. The step
        counts as idle only when every action is a no-op; ``none_counter`` is
        incremented on an idle step and reset as soon as the agent scales,
        so it measures *consecutive* inactivity.

        Args:
            action_pairs (list[tuple[int, int]]): The ``(deployment_id, action_id)``
                pairs to apply this step, as produced by the action adapter.
        """
        self.current_step += 1

        if self.current_step == 1:
            self.time_start = time.time()

        if self.current_step == self.max_steps:
            self.episode_over = True

        step_is_noop = True
        for deployment_id, action_id in action_pairs:
            self.action_stats[action_id] += 1
            self._actions[action_id].execute(self, deployment_id)
            if not self._actions[action_id].is_noop:
                step_is_noop = False

        if step_is_noop:
            self.none_counter += 1
        else:
            self.none_counter = 0

        logger.debug("[Step: %d] | Actions: %s | None counter: %d",
                     self.current_step,
                     [self._actions[a].label for _, a in action_pairs],
                     self.none_counter)

    @property
    def reward(self) -> float:
        """Returns the current reward as computed by the reward strategy."""
        return self.reward_strategy.get_reward(self)

    def get_state(self) -> np.ndarray:
        """Returns the current state of the environment.
        
        This abstract method must be implemented by subclasses to gather metrics
        from the Kubernetes cluster or simulation and construct the observation vector.

        Returns:
            numpy.ndarray: An array representing the current observation of the environment.
        """
        raise NotImplementedError

    def get_observation_space(self) -> spaces.Box:
        """Defines and returns the observation space for the environment.

        This abstract method must be implemented by subclasses to define the boundaries
        (high and low limits) and the shape of the state vector.

        Returns:
            gym.spaces.Box: The continuous multidimensional space representing the 
                valid bounds of the observations.
        """
        raise NotImplementedError

    def collect_obs(self, obs, date, latency):
        """Buffers the current step's observations into memory.

        This method should be called at every environment step. It appends
        the state and metrics to an internal list (buffer) to avoid frequent
        disk I/O.

        Args:
            obs: The observation vector from the environment state.
            date: Timestamp string for the current step.
            latency: The measured latency for the target deployment.
        """
        raise NotImplementedError

    def save_obs_to_csv(self):
        """Flushes the buffered episode observations to a CSV file.

        This method should be called once at the end of an episode. It 
        performs a bulk write of all buffered steps and should clear 
        the internal buffer upon completion.
        """
        raise NotImplementedError
