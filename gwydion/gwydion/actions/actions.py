from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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

    def execute(self, env, _deployment_id: int) -> None:
        env.none_counter += 1
        logger.debug("[Step: %d] | Action: %s | Counter: %d",
                    env.current_step, self.label, env.none_counter)

    @property
    def label(self) -> str:
        return "Do Nothing"

@dataclass
class ScaleUp(Action):
    """An action that increases the replica count of a deployment.

    Attributes:
        replicas (int): The number of additional replicas to deploy.
    """
    replicas: int

    def execute(self, env, deployment_id: int) -> None:
        deployment = env.deployment_list[deployment_id]
        constraint = deployment.deploy_pod_replicas(self.replicas)

        if constraint:
            env.constraint_max_pod_replicas = True
            logger.warning("[Step: %d] | Action: %s FAILED for %s (Limit: %s)",
                        env.current_step, self.label, deployment.name, deployment.max_pods)
        else:
            logger.debug("[Step: %d] | Action: %s for %s | Pods: %d",
                        env.current_step, self.label, deployment.name, deployment.num_pods)

    @property
    def label(self) -> str:
        return f"Scale Up {self.replicas}"

@dataclass
class ScaleDown(Action):
    """An action that decreases the replica count of a deployment.

    Attributes:
        replicas (int): The number of replicas to terminate.
    """
    replicas: int

    def execute(self, env, deployment_id: int) -> None:
        deployment = env.deployment_list[deployment_id]
        constraint = deployment.terminate_pod_replicas(self.replicas)

        if constraint:
            env.constraint_min_pod_replicas = True
            logger.warning("[Step: %d] | Action: %s FAILED for %s (Limit: %s)",
                        env.current_step, self.label, deployment.name, deployment.min_pods)
        else:
            logger.debug("[Step: %d] | Action: %s for %s | Pods: %d",
                        env.current_step, self.label, deployment.name, deployment.num_pods)

    @property
    def label(self) -> str:
        return f"Scale Down {self.replicas}"
