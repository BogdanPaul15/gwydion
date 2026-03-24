from abc import ABC, abstractmethod

class RewardStrategy(ABC):
    """Abstract base class for defining RL reward functions.

    The strategy pattern allows the environment to delegate reward calculation 
    to specialized classes. This decouples the scaling logic from the reward 
    objective (e.g., Cost vs. Latency).

    Attributes:
        penalty (float): The reward value returned when a Kubernetes constraint 
            is violated (e.g., attempting to scale below min_pods).
    """
    def __init__(self, penalty=-1.0):
        """Initializes the RewardStrategy with a configurable penalty.

        Args:
            penalty (float): The value to return upon constraint violation
                Defaults to -1.0.
        """
        self.penalty = penalty

    @abstractmethod
    def calculate(self, env):
        """Calculates the objective-specific reward based on environment state.

        Args:
            env (BaseEnv): The environment instance providing access to its
                metrics.

        Returns:
            float: The calculated reward value.
        """

    def get_reward(self, env):
        """Main entry point for the environment to fetch the current reward.

        This method handles the high-level logic of checking for boundary
        constraints before proceeding to the specific reward calculation.

        Args:
            env (BaseEnv): The environment instance to check for constraints.

        Returns:
            float: The constraint penalty if limits were hit, otherwise the
                result of the calculate method.
        """
        if env.constraint_min_pod_replicas or env.constraint_max_pod_replicas:
            return self.get_constraint_penalty()

        return self.calculate(env)

    def get_constraint_penalty(self):
        """Returns the fixed penalty value defined during initialization.

        Returns:
            float: The penalty value.
        """
        return self.penalty

class CostStrategy(RewardStrategy):
    """Reward strategy focused on infrastructure cost and deployment stability.

    This strategy rewards the agent based on how many deployments have
    the correct number of running pods relative to the desired replica count.
    It penalizes inactivity (none_counter) when the cluster is not in the 
    desired state, forcing the agent to avoid sub-optimal states.
    """
    def calculate(self, env):
        reward = sum(1 for d in env.deployment_list if d.num_pods == d.desired_replicas)

        if reward != env.num_apps and env.none_counter > 2:
            reward = -env.none_counter
        return reward

class LatencyStrategy(RewardStrategy):
    """Reward strategy focused on application performance and response time.

    This strategy penalizes the agent based on the latency of a specific
    target deployment. It applies a ceiling (threshold) to the penalty to
    prevent extreme values from destabilizing the learning process.

    Attributes:
        target_id (int): Index of the target deployment to monitor.
        threshold (float): The maximum latency allowed.
        penalty (float): Penalty for constraint violations.
    """
    def __init__(self, target_id, threshold, penalty=None):
        penalty = penalty if penalty is not None else -threshold
        super().__init__(penalty=penalty)

        self.target_id = target_id
        self.threshold = threshold

    def calculate(self, env):
<<<<<<< HEAD
        latency = float(env.deployment_list[self.target_id].latency)
=======
        latency = float(env.deploymentList[self.target_id].metrics["latency"])
>>>>>>> 1ce298f (fix: solve reward bugs)

        reward = -min(latency, self.threshold)

        if env.none_counter > 2:
            reward = -self.threshold * env.none_counter
        return reward
