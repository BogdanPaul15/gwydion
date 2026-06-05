from typing import Any
import optuna

from .maps import ACTIVATION_FN_MAP, NET_ARCH_MAP

def sample_trpo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample TRPO hyperparameters for one Optuna trial."""
    n_steps_pow    = trial.suggest_int("n_steps_pow", 5, 7)  # 32 to 128
    batch_size_pow = trial.suggest_int("batch_size_pow", 4, 6)  # 16 to 64

    one_minus_gamma      = trial.suggest_float("one_minus_gamma", 0.01, 0.2, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate    = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)

    n_critic_updates = trial.suggest_int("n_critic_updates", 5, 30)
    cg_max_steps     = trial.suggest_int("cg_max_steps", 5, 30)
    target_kl        = trial.suggest_float("target_kl", 0.001, 0.1, log=True)
    net_arch         = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn    = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma",      1 - one_minus_gamma)
    trial.set_user_attr("gae_lambda", 1 - one_minus_gae_lambda)
    trial.set_user_attr("n_steps",    2 ** n_steps_pow)
    trial.set_user_attr("batch_size", 2 ** batch_size_pow)

    return {
        "n_steps_pow":           n_steps_pow,
        "batch_size_pow":        batch_size_pow,
        "one_minus_gamma":       one_minus_gamma,
        "one_minus_gae_lambda":  one_minus_gae_lambda,
        "learning_rate":         learning_rate,
        "n_critic_updates":      n_critic_updates,
        "cg_max_steps":          cg_max_steps,
        "target_kl":             target_kl,
        "net_arch":              net_arch,
        "activation_fn":         activation_fn,
    }

def convert_trpo_params(sampled: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Translate raw sample_trpo_params() dict into TRPO(**kwargs)."""
    hyperparams = sampled.copy()

    n_steps = 2 ** hyperparams.pop("n_steps_pow")
    batch_size = 2 ** hyperparams.pop("batch_size_pow")

    rollout_size = n_steps * n_envs
    batch_size = min(batch_size, rollout_size)

    # Ensure batch_size divides rollout buffer evenly
    while rollout_size % batch_size != 0:
        batch_size //= 2

    hyperparams["n_steps"] = n_steps
    hyperparams["batch_size"] = batch_size

    hyperparams["gamma"] = 1 - hyperparams.pop("one_minus_gamma")
    hyperparams["gae_lambda"] = 1 - hyperparams.pop("one_minus_gae_lambda")

    hyperparams["policy_kwargs"] = {
        "net_arch": NET_ARCH_MAP[hyperparams.pop("net_arch")],
        "activation_fn": ACTIVATION_FN_MAP[hyperparams.pop("activation_fn")],
    }

    return hyperparams
