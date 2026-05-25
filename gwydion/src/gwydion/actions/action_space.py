from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from gymnasium import spaces

DecodedAction = List[Tuple[int, int]]

class ActionSpaceAdapter(ABC):
    """Translates between a gym action space and per-deployment scaling commands.

    ``decode`` always returns a *list* of ``(deployment_id, action_id)`` pairs:
    single-target adapters yield a one-element list (one deployment scaled per
    step), while :class:`VectorAdapter` yields one pair per deployment so
    every deployment can be scaled simultaneously.
    """

    def __init__(self, num_deployments: int, num_actions: int):
        self.num_deployments = num_deployments
        self.num_actions = num_actions

    @property
    @abstractmethod
    def gym_space(self) -> spaces.Space:
        """The Gymnasium space exposed to the RL algorithm."""

    @abstractmethod
    def decode(self, raw_action) -> DecodedAction:
        """Converts the raw RL action into a list of (deployment_id, action_id) pairs.

        Args:
            raw_action: Whatever the RL algorithm outputs (int, np.ndarray, etc.)

        Returns:
            DecodedAction: One ``(deployment_id, action_id)`` pair per deployment
                to be scaled this step.
        """

    @abstractmethod
    def encode(self, actions: DecodedAction):
        """Converts a list of (deployment_id, action_id) pairs into the raw format.

        Args:
            actions (DecodedAction): The decoded action pairs.

        Returns:
            The raw action in the format expected by :attr:`gym_space`.
        """

class MultiDiscreteAdapter(ActionSpaceAdapter):
    """Two-dimensional action: ``[which_deployment, which_action]``.

    Exactly one deployment is scaled per step.
    """

    @property
    def gym_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([self.num_deployments, self.num_actions])

    def decode(self, raw_action) -> DecodedAction:
        return [(int(raw_action[0]), int(raw_action[1]))]

    def encode(self, actions: DecodedAction):
        deployment_id, action_id = actions[0]
        return np.array([deployment_id, action_id])

class DiscreteAdapter(ActionSpaceAdapter):
    """Flattened single-integer action; exactly one deployment scaled per step.

    Maps the 2D (deployment, action) space into a 1D integer:
    ``flat_id = deployment_id * num_actions + action_id``.
    """

    @property
    def gym_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_deployments * self.num_actions)

    def decode(self, raw_action) -> DecodedAction:
        raw = int(raw_action)
        return [(raw // self.num_actions, raw % self.num_actions)]

    def encode(self, actions: DecodedAction):
        deployment_id, action_id = actions[0]
        return deployment_id * self.num_actions + action_id

class VectorAdapter(ActionSpaceAdapter):
    """One scaling action per deployment, applied simultaneously each step.

    The action is a vector of ``num_deployments`` action ids — entry ``i`` is the
    scaling action for deployment ``i``. This lets the agent correct every
    deployment in a single step instead of being limited to one target.
    """

    @property
    def gym_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([self.num_actions] * self.num_deployments)

    def decode(self, raw_action) -> DecodedAction:
        return [(i, int(raw_action[i])) for i in range(self.num_deployments)]

    def encode(self, actions: DecodedAction):
        ordered = sorted(actions, key=lambda pair: pair[0])
        return np.array([action_id for _, action_id in ordered])

_ADAPTERS = {
    "multi_discrete": MultiDiscreteAdapter,
    "discrete": DiscreteAdapter,
    "vector": VectorAdapter,
}

def build_action_space(space_type: str, num_deployments: int,
                       num_actions: int) -> ActionSpaceAdapter:
    """Factory that creates the appropriate adapter from a config string.

    Args:
        space_type: One of "multi_discrete", "discrete", or "vector".
        num_deployments: Number of deployments the agent can target.
        num_actions: Number of scaling actions available.

    Returns:
        An ActionSpaceAdapter instance.

    Raises:
        ValueError: If space_type is not recognized.
    """
    if space_type not in _ADAPTERS:
        raise ValueError(
            f"Unknown action space type '{space_type}'. "
            f"Available: {list(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[space_type](num_deployments, num_actions)
