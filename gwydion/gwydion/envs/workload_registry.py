from __future__ import annotations
from typing import Type
from gwydion.envs.workload import BaseDeploymentWorkload

_REGISTRY: dict[str, Type[BaseDeploymentWorkload]] = {}

def register(name: str):
    """Class decorator that registers a workload type by name.

    Usage:
        @register("redis")
        class RedisWorkload(BaseDeploymentWorkload):
            ...

    Args:
        name (str): The key used to look up this workload from config.
    """
    def decorator(cls: Type[BaseDeploymentWorkload]):
        if name in _REGISTRY:
            raise ValueError(
                f"Workload type '{name}' is already registered "
                f"by {_REGISTRY[name].__name__}. "
                f"Each workload type must have a unique name."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator

def build_workload(cfg: dict, k8s: bool) -> BaseDeploymentWorkload:
    """Instantiates a workload from a config dict using the registry.

    The config dict must contain a 'type' key matching a registered name.
    All other keys are forwarded to the workload's from_config() classmethod.

    Args:
        cfg (dict): Workload configuration dictionary.
        k8s (bool): Whether to connect to a real K8s cluster.

    Returns:
        BaseDeploymentWorkload: A fully initialized workload instance.

    Raises:
        ValueError: If the workload type is not registered.
    """
    workload_type = cfg["type"]
    cls = _REGISTRY.get(workload_type)

    if cls is None:
        registered = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown workload type: '{workload_type}'."
            f"Registered types are: {registered}"
        )

    return cls.from_config(cfg, k8s)

def build_deployment_list(deployment_cfgs: list[dict], k8s: bool) -> list[BaseDeploymentWorkload]:
    """Builds a full deployment list from a list of config dicts.

    Args:
        deployment_cfgs (list[dict]): List of per-deployment config dicts.
        k8s (bool): Whether to connect to a real K8s cluster.

    Returns:
        list[BaseDeploymentWorkload]: Ordered list of workload instances.
    """
    return [build_workload(cfg, k8s) for cfg in deployment_cfgs]

def list_registered() -> list[str]:
    """Returns all currently registered workload type names.
    Useful for debugging and validation.
    """
    return list(_REGISTRY.keys())
