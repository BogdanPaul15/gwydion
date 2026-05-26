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

    def __init__(self, penalty=-1.0) -> None:
        """Initializes the strategy with a configurable penalty.

        Args:
            penalty (float): The value to return upon constraint violation
                Defaults to -1.0.
        """
        self.penalty = penalty

    @abstractmethod
    def calculate(self, env) -> float:
        """Calculates the objective-specific reward based on environment state.

        Args:
            env (BaseEnv): The environment instance providing access to its
                metrics.

        Returns:
            float: The calculated reward value.
        """

    def get_reward(self, env) -> float:
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

    def get_constraint_penalty(self) -> float:
        """Returns the fixed penalty value defined during initialization.

        Returns:
            float: The penalty value.
        """
        return self.penalty

class CostStrategy(RewardStrategy):
    """Reward strategy focused on infrastructure cost and deployment stability.

    This strategy rewards the agent based on how many deployments have
    the correct number of running pods relative to the desired replica count.
    It penalizes inactivity (``none_counter``) when the cluster is not in the 
    desired state, forcing the agent to avoid sub-optimal states.
    """

    def calculate(self, env):
        reward = sum(1 for d in env.deployment_list if d.num_pods == d.desired_replicas)

        if reward != env.num_apps and env.none_counter > 2:
            reward = -env.none_counter
        return reward

class SmoothCostStrategy(RewardStrategy):
    """Cost reward with a tolerance band and smooth partial credit.

    Extends :class:`CostStrategy` and provides a more informative reward
    signal that guides the agent towards the target pod counts.

    A deployment earns full credit when its pod count is within 
    ``tolerance`` of ``desired_replicas``.
    Deployments outside the band still receive smooth partial credit
    ``1 / (1 + error - tolerance)`` that decreases with the
    distance to the target, giving the agent a gradient to follow instead of
    a sparse all-or-nothing signal.

    Attributes:
        tolerance (int): Pod-count deviation still counted as a full match.
        patience (int): Consecutive noop steps allowed before penalizing
            inactivity while the cluster is outside the band.
    """

    def __init__(self, tolerance: int = 1, patience: int = 2, penalty: float = -1.0):
        """Initializes the shaped cost strategy.

        Args:
            tolerance (int): Pod-count deviation still scored as a full match.
            patience (int): Consecutive noop steps allowed before penalizing
                inactivity while the cluster is mis-scaled.
            penalty (float): The reward value returned when a Kubernetes constraint 
                is violated (e.g., attempting to scale below ``min_pods``).
        """
        super().__init__(penalty=penalty)
        self.tolerance = tolerance
        self.patience = patience

    def calculate(self, env):
        reward = 0.0
        needs_scaling = False

        for d in env.deployment_list:
            error = abs(d.num_pods - d.desired_replicas)
            if error <= self.tolerance:
                reward += 1.0
            else:
                reward += 1.0 / (1.0 + error - self.tolerance)
                needs_scaling = True

        if needs_scaling and env.none_counter > self.patience:
            reward = -float(env.none_counter)
        return reward

class MultiObjectiveStrategy(RewardStrategy):
    """Weighted linear combination of multiple reward objectives.

    Each objective contributes ``weight * strategy.calculate(env)`` to the
    final reward.  The objectives are evaluated independently so their
    individual scale differences are preserved — choose weights that reflect
    both importance and scale (e.g. a latency term in seconds needs a much
    larger weight than a cost term already in [0, num_apps]).

    Constraint violations return this strategy's own ``penalty`` directly,
    bypassing all objectives.

    Attributes:
        objectives (list[tuple[RewardStrategy, float]]): Ordered pairs of
            ``(strategy, weight)``.
    """

    def __init__(self, objectives: list, penalty: float = -1.0):
        """Initialises the multi-objective combinator.

        Args:
            objectives (list[tuple[RewardStrategy, float]]): Each element is a
                ``(strategy, weight)`` pair.  Weights need not sum to 1.
            penalty (float): The reward value returned when a Kubernetes constraint 
                is violated (e.g., attempting to scale below ``min_pods``).
        """
        super().__init__(penalty=penalty)
        self.objectives = objectives

    def calculate(self, env) -> float:
        return sum(
            weight * strategy.calculate(env)
            for strategy, weight in self.objectives
        )

class LatencyStrategy(RewardStrategy):
    """Reward strategy focused on application performance and response time.

    This strategy penalizes the agent based on the latency of a specific
    target deployment. It applies a ceiling (threshold) to the penalty to
    prevent extreme values from destabilizing the learning process.

    Attributes:
        target_id (int): Index of the target deployment to monitor.
        threshold (float): The maximum latency allowed.
        penalty (float): The reward value returned when a Kubernetes constraint 
            is violated (e.g., attempting to scale below ``min_pods``).
    """

    def __init__(self, target_id, threshold, penalty=None):
        penalty = penalty if penalty is not None else -threshold
        super().__init__(penalty=penalty)

        self.target_id = target_id
        self.threshold = threshold

    def calculate(self, env):
        latency = float(env.deployment_list[self.target_id].metrics["latency"])

        reward = -min(latency, self.threshold)

        if env.none_counter > 2:
            reward = -self.threshold * env.none_counter
        return reward
