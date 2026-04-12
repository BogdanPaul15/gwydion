from typing import Any
import optuna

from .maps import ACTIVATION_FN_MAP, NET_ARCH_MAP

def linear_schedule(initial_value: float):
    """Decays learning rate linearly from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def sample_ppo_params(trial: optuna.Trial, n_envs: int = 1) -> dict:
    """Sample PPO hyperparameters for one Optuna trial."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10) # 4 to 1024
    n_steps_pow    = trial.suggest_int("n_steps_pow", 5, 12) # 32 to 4096

    # Discount factor - sampled as (1 - gamma) in log-scale
    one_minus_gamma      = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    lr_schedule   = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
    ent_coef      = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range    = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs      = trial.suggest_int("n_epochs", 5, 20)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)

    net_arch      = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    n_steps    = 2 ** n_steps_pow
    batch_size = 2 ** batch_size_pow
    if batch_size > n_steps * n_envs:
        batch_size_pow = n_steps_pow
        batch_size     = n_steps

    trial.set_user_attr("gamma",      1 - one_minus_gamma)
    trial.set_user_attr("gae_lambda", 1 - one_minus_gae_lambda)
    trial.set_user_attr("n_steps",    n_steps)
    trial.set_user_attr("batch_size", batch_size)

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

def convert_ppo_params(sampled: dict[str, Any]) -> dict[str, Any]:
    """Translate raw sample_ppo_params() dict into PPO(**kwargs)."""
    hyperparams = sampled.copy()

    if "batch_size_pow" in hyperparams:
        hyperparams["batch_size"] = 2 ** hyperparams.pop("batch_size_pow")
    if "n_steps_pow" in hyperparams:
        hyperparams["n_steps"] = 2 ** hyperparams.pop("n_steps_pow")

    if "one_minus_gamma" in hyperparams:
        hyperparams["gamma"] = 1 - hyperparams.pop("one_minus_gamma")
    if "one_minus_gae_lambda" in hyperparams:
        hyperparams["gae_lambda"] = 1 - hyperparams.pop("one_minus_gae_lambda")

    lr_schedule = hyperparams.pop("lr_schedule", "constant")
    if lr_schedule == "linear":
        hyperparams["learning_rate"] = linear_schedule(hyperparams["learning_rate"])

    net_arch = hyperparams.pop("net_arch", None)
    activation_fn = hyperparams.pop("activation_fn", None)

    if net_arch or activation_fn:
        policy_kwargs = {}
        if net_arch:
            policy_kwargs["net_arch"] = NET_ARCH_MAP[net_arch]
        if activation_fn:
            policy_kwargs["activation_fn"] = ACTIVATION_FN_MAP[activation_fn]
        hyperparams["policy_kwargs"] = policy_kwargs

    return hyperparams
