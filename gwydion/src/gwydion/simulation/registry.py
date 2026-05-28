from typing import Type, Any, Optional
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

def build_simulation_strategies(cfg: dict, seed: Optional[int] = None, **kwargs: Any) -> SimulationStrategy:
    """Instantiates a simulation strategy from a config dict.

    Args:
        cfg: Must contain a ``type`` key matching a name registered via
            :func:`register`. Remaining keys are passed to the strategy's
            ``config`` parameter.
        seed: Optional random seed forwarded to ``strategy.seed()`` after
            construction.
        **kwargs: Extra keyword arguments forwarded to the strategy constructor
            (e.g. ``df``, ``deployment_names``).

    Returns:
        SimulationStrategy: The constructed strategy, already seeded.

    Raises:
        ValueError: If ``cfg["type"]`` is not registered.
    """
    strategy_type = cfg["type"]
    cls = _REGISTRY.get(strategy_type)

    if cls is None:
        registered = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy type: '{strategy_type}'. "
            f"Registered strategies are: {registered}"
        )

    strategy = cls(config=cfg, **kwargs)
    strategy.seed(seed)

    return strategy
