from typing import List
from gwydion.envs.actions import Action, DoNothing, ScaleUp, ScaleDown

_ACTION = {
    "do_nothing":    lambda _: DoNothing(),
    "scale_up":      lambda cfg: ScaleUp(replicas=cfg["replicas"]),
    "scale_down":    lambda cfg: ScaleDown(replicas=cfg["replicas"]),
}

def build_action_set(action_cfgs: list) -> List[Action]:
    """Builds a list of Action objects from a config.

    This function maps the YAML configuration actions to concrete Action objects,
    allowing the action space to be defined entirely via config file.

    Args:
        action_cfgs(list): A list of dictionaries from the YAML configuration.

    Returns:
        List[Action]: An ordered list of instantiated Action objects.

    Raises:
        ValueError: If an unknown action type is encountered in the config.
    """
    actions = []
    for cfg in action_cfgs:
        action_type = cfg["type"]
        builder = _ACTION.get(action_type)
        if builder is None:
            raise ValueError(f"Unknown action type: '{action_type}'")
        actions.append(builder(cfg))
    return actions
