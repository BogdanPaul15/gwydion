from typing import List, Optional

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

MIN_REPLICATION = 1
MAX_REPLICATION = 8
MAX_STEPS = 25
ACTION_MOVES_COUNT = 15

COST = "cost"
LATENCY = "latency"

class BaseEnv(gym.Env):
    """Abstract Base Class for Kubernetes Horizontal Scaling Environments.
    
    This class provides a common interface and shared logic for Reinforcement Learning
    environments controlling pod replication in a K8s cluster or simulation.

    Attributes:
        name (str): The unique name of the environment.
        num_apps (int): The number of managed deployments.
        deployments_name (list[str]): Name of the K8s deployments.
        k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
        goal_reward (str): The reward objective function.
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
        deploymentList (List[DeploymentStatus]): A list of DeploymentStatus objects representing
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
                 k8s: bool = False, goal_reward: str = COST, waiting_period: int = 5):
        """Initializes the BaseEnv with scaling constraints and core attributes.

        Args:
            name (str): The unique name of the environment.
            num_apps (int): The number of managed deployments.
            k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
            goal_reward (str): The reward objective function.
            waiting_period (int): Seconds to wait after a scaling (real K8s only).

        """
        super(BaseEnv, self).__init__()

        self.name = name
        self.num_apps = num_apps
        self.deployments_names = deployments
        self.k8s = k8s
        self.goal_reward = goal_reward
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

        # TODO: refactor
        self.num_actions = ACTION_MOVES_COUNT
        self.none_counter = 0
        self.action_stats = [0 for _ in range(self.num_actions)]
        self.traffic = []

        self.avg_pods = []
        self.avg_latency = []
        self.time_start = 0
        self.execution_time = 0

        # TODO: rename the attribute
        self.deploymentList = []
        self.action_space = spaces.MultiDiscrete([num_apps, self.num_actions])
        self.observation_space = None

        self.df = None

    def load_dataset(self):
        """Loads the simulation dataframe using deployment metadata.
        
        This must be called AFTER self.deploymentList is initialized in the child class.
        """
        if not self.k8s:
            # Get namespace from the first deployment in the list
            namespace = self.deploymentList[0].namespace
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

        # TODO self.none_counter should be added here as well
        self.current_step = 0
        self.total_reward = 0

        self.terminated = False
        self.episode_over = False
        self.constraint_min_pod_replicas = False
        self.constraint_max_pod_replicas = False

        self.avg_pods = []
        self.avg_latency = []

        # TODO should also reset self.action_stats, self.time_start, self.execution_time, self.info

        # Note: self.deploymentList should be reinitialized in the child
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
        raise NotImplementedError

    def take_action(self, action, id):
        raise NotImplementedError

    def simulation_update(self, action):
        raise NotImplementedError

    def calculate_reward(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError
