from typing import Any
import optuna

from .ppo import sample_ppo_params, convert_ppo_params

def sample_maskable_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample MaskablePPO hyperparameters for one Optuna trial."""
    return sample_ppo_params(trial)

def convert_maskable_ppo_params(sampled: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Translate raw sample_maskable_ppo_params() dict into MaskablePPO(**kwargs)."""
    return convert_ppo_params(sampled, n_envs)
