from typing import Any
import optuna

from .ppo import sample_ppo_params, convert_ppo_params

def sample_recurrent_ppo_params(trial: optuna.Trial) -> dict:
    """Sample RecurrentPPO hyperparameters for one Optuna trial."""
    sampled = sample_ppo_params(trial)

    sampled.update({
        "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128, 256, 512]),
        "n_lstm_layers": trial.suggest_categorical("n_lstm_layers", [1, 2]),
        "enable_critic_lstm": trial.suggest_categorical("enable_critic_lstm", [True, False]),
    })

    return sampled

def convert_recurrent_ppo_params(sampled: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Translate raw sample_recurrent_ppo_params() dict into RecurrentPPO(**kwargs)."""
    lstm_hidden_size   = sampled.pop("lstm_hidden_size")
    n_lstm_layers      = sampled.pop("n_lstm_layers")
    enable_critic_lstm = sampled.pop("enable_critic_lstm")

    hyperparams = convert_ppo_params(sampled, n_envs)

    hyperparams["policy_kwargs"].update({
        "lstm_hidden_size": int(lstm_hidden_size),
        "n_lstm_layers": int(n_lstm_layers),
        "enable_critic_lstm": bool(enable_critic_lstm),
    })

    return hyperparams
