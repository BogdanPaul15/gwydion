from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from gymnasium import spaces

class ActionSpaceAdapter(ABC):
    """Translates between a gym action space and (deployment_id, action_id) tuples."""

    def __init__(self, num_deployments: int, num_actions: int):
        self.num_deployments = num_deployments
        self.num_actions = num_actions

    @property
    @abstractmethod
    def gym_space(self) -> spaces.Space:
        """The Gymnasium space exposed to the RL algorithm."""

    @abstractmethod
    def decode(self, raw_action) -> Tuple[int, int]:
        """Converts the raw action from the RL algorithm into (deployment_id, action_id).
        
        Args:
            raw_action: Whatever the RL algorithm outputs (int, np.ndarray, etc.)
        
        Returns:
            Tuple of (deployment_id, action_id).
        """

    @abstractmethod
    def encode(self, deployment_id: int, action_id: int):
        """Converts (deployment_id, action_id) back into the raw action format.
        
        Args:
            deployment_id: Index of the target deployment.
            action_id: Index of the scaling action.

        Returns:
            The raw action in the format expected by the gym space.
        """

class MultiDiscreteAdapter(ActionSpaceAdapter):
    """Two-dimensional action: [which_deployment, which_action]"""

    @property
    def gym_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([self.num_deployments, self.num_actions])

    def decode(self, raw_action) -> Tuple[int, int]:
        return int(raw_action[0]), int(raw_action[1])

    def encode(self, deployment_id: int, action_id: int):
        return np.array([deployment_id, action_id])

class DiscreteAdapter(ActionSpaceAdapter):
    """Flattened action: single  integer encoding.
    
    Maps the 2D (deployment, action) space into a 1D integer:
    flat_id = deployment_id * num_actions + action_id
    """

    @property
    def gym_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_deployments * self.num_actions)

    def decode(self, raw_action) -> Tuple[int, int]:
        raw = int(raw_action)
        deployment_id = raw // self.num_actions
        action_id = raw % self.num_actions
        return deployment_id, action_id

    def encode(self, deployment_id: int, action_id: int):
        return deployment_id * self.num_actions + action_id

_ADAPTERS = {
    "multi_discrete": MultiDiscreteAdapter,
    "discrete": DiscreteAdapter,
}

def build_action_space(space_type: str, num_deployments: int,
                       num_actions: int) -> ActionSpaceAdapter:
    """Factory that creates the appropriate adapter from a config string.

    Args:
        space_type: One of "multi_discrete", "discrete".
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
