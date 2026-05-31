from typing import Any
import optuna

from .maps import ACTIVATION_FN_MAP, NET_ARCH_MAP

def linear_schedule(initial_value: float):
    """Decays learning rate linearly from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sample PPO hyperparameters for one Optuna trial."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 4, 6) # 32 to 256
    n_steps_pow    = trial.suggest_int("n_steps_pow", 5, 7) # 256 to 1024

    # Discount factor - sampled as (1 - gamma) in log-scale
    one_minus_gamma      = trial.suggest_float("one_minus_gamma", 0.01, 0.2, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    lr_schedule   = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
    ent_coef      = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    clip_range    = trial.suggest_categorical("clip_range", [0.2, 0.3])
    n_epochs      = trial.suggest_int("n_epochs", 5, 15)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    net_arch      = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma",      1 - one_minus_gamma)
    trial.set_user_attr("gae_lambda", 1 - one_minus_gae_lambda)
    trial.set_user_attr("n_steps",    2 ** n_steps_pow)
    trial.set_user_attr("batch_size", 2 ** batch_size_pow)

    return {
        "batch_size_pow":       batch_size_pow,
        "n_steps_pow":          n_steps_pow,
        "one_minus_gamma":      one_minus_gamma,
        "one_minus_gae_lambda": one_minus_gae_lambda,
        "learning_rate":        learning_rate,
        "lr_schedule":          lr_schedule,
        "ent_coef":             ent_coef,
        "clip_range":           clip_range,
        "n_epochs":             n_epochs,
        "max_grad_norm":        max_grad_norm,
        "net_arch":             net_arch,
        "activation_fn":        activation_fn,
    }

def convert_ppo_params(sampled: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Translate raw sample_ppo_params() dict into PPO(**kwargs)."""
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

    lr_schedule = hyperparams.pop("lr_schedule", "constant")
    if lr_schedule == "linear":
        hyperparams["learning_rate"] = linear_schedule(hyperparams["learning_rate"])

    hyperparams["policy_kwargs"] = {
        "net_arch": NET_ARCH_MAP[hyperparams.pop("net_arch")],
        "activation_fn": ACTIVATION_FN_MAP[hyperparams.pop("activation_fn")],
    }

    return hyperparams
