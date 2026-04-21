from typing import Type, Any
from .base import SimulationStrategy

_REGISTRY: dict[str, Type[SimulationStrategy]] = {}

def register(name: str):
    """Class decorator that registers a simulation strategy type by name.

    Usage:
        @register("default")
        class DefaultSimulationStrategy(SimulationStrategy):
            ...
    
    Args:
        name (str): The key used to look up this simulation strategy from config.
    """
    def decorator(cls: Type[SimulationStrategy]):
        if name in _REGISTRY:
            raise ValueError(
                f"Simulation strategy type '{name}' is already registered "
                f"by {_REGISTRY[name].__name__}. "
                f"Each simulation strategy type must have a unique name."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator

def build_simulation_strategies(cfg: dict, **kwargs: Any) -> SimulationStrategy:
    """TODO"""
    strategy_type = cfg["type"]
    cls = _REGISTRY.get(strategy_type)

    if cls is None:
        registered = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy type: '{strategy_type}'. "
            f"Registered strategies are: {registered}"
        )

    return cls(config=cfg, **kwargs)
