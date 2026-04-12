from dataclasses import dataclass
from typing import Any, Callable, Type

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import TRPO, MaskablePPO, RecurrentPPO

from gwydion.arena.hyperparams.ppo import sample_ppo_params, convert_ppo_params
from gwydion.arena.hyperparams.recurrent_ppo import sample_recurrent_ppo_params, convert_recurrent_ppo_params
from gwydion.arena.hyperparams.maskable_ppo import sample_maskable_ppo_params, convert_maskable_ppo_params
from gwydion.arena.hyperparams.trpo import sample_trpo_params, convert_trpo_params
from gwydion.arena.hyperparams.a2c import sample_a2c_params, convert_a2c_params


@dataclass(frozen=True)
class AlgorithmSpec:
    """All metadata Arena needs to tune, train, and evaluate one algorithm.
    
    Attributes:
        cls: The SB3 algorithm class.
        policy: The SB3 policy string.
        sampler: Sampling method for SB3 algorithm hyperparams. 
        converter: Converter from Optuna hyperparams suggestions to SB3 hyperparams.
    """
    cls:                  Type[BaseAlgorithm]
    policy:               str
    sampler:              Callable[..., dict[str, Any]]
    converter:            Callable[[dict[str, Any]], dict[str, Any]]

ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "ppo": AlgorithmSpec(
        cls=PPO,
        policy="MlpPolicy",
        sampler=sample_ppo_params,
        converter=convert_ppo_params,
    ),
    "recurrent_ppo": AlgorithmSpec(
        cls=RecurrentPPO,
        policy="MlpLstmPolicy",
        sampler=sample_recurrent_ppo_params,
        converter=convert_recurrent_ppo_params,
    ),
    "maskable_ppo": AlgorithmSpec(
        cls=MaskablePPO,
        policy="MlpPolicy",
        sampler=sample_maskable_ppo_params,
        converter=convert_maskable_ppo_params,
    ),
    "trpo": AlgorithmSpec(
        cls=TRPO,
        policy="MlpPolicy",
        sampler=sample_trpo_params,
        converter=convert_trpo_params,
    ),
    "a2c": AlgorithmSpec(
        cls=A2C,
        policy="MlpPolicy",
        sampler=sample_a2c_params,
        converter=convert_a2c_params,
    ),
}

def get_spec(alg: str) -> AlgorithmSpec:
    """Return the AlgorithmSpec for the given algorithm key.
 
    Raises:
        ValueError: If the algorithm is not registered.
    """
    alg = alg.lower()
    if alg not in ALGORITHM_SPECS:
        raise ValueError(
            f"Unknown algorithm '{alg}'. "
            f"Available: {sorted(ALGORITHM_SPECS)}."
        )
    return ALGORITHM_SPECS[alg]
