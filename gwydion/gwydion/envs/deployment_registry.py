from __future__ import annotations
from typing import Type
from gwydion.gwydion.envs.deployment import Deployment

_REGISTRY: dict[str, Type[Deployment]] = {}

def register(name: str):
    """Class decorator that registers a deployment type by name.

    Usage:
        @register("redis")
        class RedisDeployment(Deployment):
            ...

    Args:
        name (str): The key used to look up this deployment from config.
    """
    def decorator(cls: Type[Deployment]):
        if name in _REGISTRY:
            raise ValueError(
                f"Deployment type '{name}' is already registered "
                f"by {_REGISTRY[name].__name__}. "
                f"Each deployment type must have a unique name."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator

def build_deployment(cfg: dict, k8s: bool) -> Deployment:
    """Instantiates a deployment from a config dict using the registry.

    The config dict must contain a 'type' key matching a registered name.
    All other keys are forwarded to the deployment's from_config() classmethod.

    Args:
        cfg (dict): Deployment configuration dictionary.
        k8s (bool): Whether to connect to a real K8s cluster.

    Returns:
        Deployment: A fully initialized deployment instance.

    Raises:
        ValueError: If the deployment type is not registered.
    """
    deployment_type = cfg["type"]
    cls = _REGISTRY.get(deployment_type)

    if cls is None:
        registered = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown deployment type: '{deployment_type}'."
            f"Registered types are: {registered}"
        )

    return cls.from_config(cfg, k8s)

def build_deployment_list(deployment_cfgs: list[dict], k8s: bool) -> list[Deployment]:
    """Builds a full deployment list from a list of config dicts.

    Args:
        deployment_cfgs (list[dict]): List of per-deployment config dicts.
        k8s (bool): Whether to connect to a real K8s cluster.

    Returns:
        list[Deployment]: Ordered list of deployment instances.
    """
    return [build_deployment(cfg, k8s) for cfg in deployment_cfgs]

def list_registered() -> list[str]:
    """Returns all currently registered deployment type names.
    Useful for debugging and validation.
    """
    return list(_REGISTRY.keys())
