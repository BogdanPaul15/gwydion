from abc import ABC, abstractmethod
from dataclasses import dataclass

class Action(ABC):
    """Abstract Base Class for deployment scaling actions.

    The Action class encapsulates all the logic required to perform
    a specific scaling operation on a Kubernetes deployment.
    """

    @abstractmethod
    def execute(self, env, deployment_id: int) -> None:
        """Executes the scaling logic against a specific deployment in the environment.
        
        Args:
            env (BaseEnv): The active Gymansium environment instance.
            deployment_id (int): The index of the target deployment.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        """Returns a string representation of the action."""
        raise NotImplementedError

@dataclass
class DoNothing(Action):
    """An action that maintains the current state without scaling."""

    def execute(self, env, deployment_id: int) -> None:
        env.none_counter += 1

    @property
    def label(self) -> str:
        return "DO_NOTHING"

@dataclass
class ScaleUp(Action):
    """An action that increases the replica count of a deployment.

    Attributes:
        replicas (int): The number of additional replicas to deploy.
    """
    replicas: int

    def execute(self, env, deployment_id: int) -> None:
        env.deployment_list[deployment_id].deploy_pod_replicas(self.replicas, env)

    @property
    def label(self) -> str:
        return f"SCALE_UP_{self.replicas}"

@dataclass
class ScaleDown(Action):
    """An action that decreases the replica count of a deployment.

    Attributes:
        replicas (int): The number of replicas to terminate.
    """
    replicas: int

    def execute(self, env, deployment_id: int) -> None:
        env.deployment_list[deployment_id].terminate_pod_replicas(self.replicas, env)

    @property
    def label(self) -> str:
        return f"SCALE_DOWN_{self.replicas}"
