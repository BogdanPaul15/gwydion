from typing import Any
import optuna

from .maps import ACTIVATION_FN_MAP, NET_ARCH_MAP

def sample_trpo_params(trial: optuna.Trial, n_envs: int = 1) -> dict[str, Any]:
    """Sample TRPO hyperparameters for one Optuna trial."""
    n_steps_pow    = trial.suggest_int("n_steps_pow", 5, 12)
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)

    one_minus_gamma      = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate    = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef         = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    max_grad_norm    = trial.suggest_float("max_grad_norm", 0.3, 2.0)

    n_critic_updates = trial.suggest_int("n_critic_updates", 5, 30)
    cg_max_steps     = trial.suggest_int("cg_max_steps", 5, 30)
    target_kl        = trial.suggest_float("target_kl", 0.001, 0.1, log=True)
    net_arch         = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn    = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

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
        "n_steps_pow":           n_steps_pow,
        "batch_size_pow":        batch_size_pow,
        "one_minus_gamma":       one_minus_gamma,
        "one_minus_gae_lambda":  one_minus_gae_lambda,
        "learning_rate":         learning_rate,
        "ent_coef":              ent_coef,
        "max_grad_norm":         max_grad_norm,
        "n_critic_updates":      n_critic_updates,
        "cg_max_steps":          cg_max_steps,
        "target_kl":             target_kl,
        "net_arch":              net_arch,
        "activation_fn":         activation_fn,
    }

def convert_trpo_params(sampled: dict[str, Any]) -> dict[str, Any]:
    """Translate raw sample_trpo_params() dict into TRPO(**kwargs)."""
    hyperparams = sampled.copy()

    if "batch_size_pow" in hyperparams:
        hyperparams["batch_size"] = 2 ** hyperparams.pop("batch_size_pow")
    if "n_steps_pow" in hyperparams:
        hyperparams["n_steps"] = 2 ** hyperparams.pop("n_steps_pow")

    if "one_minus_gamma" in hyperparams:
        hyperparams["gamma"] = 1 - hyperparams.pop("one_minus_gamma")
    if "one_minus_gae_lambda" in hyperparams:
        hyperparams["gae_lambda"] = 1 - hyperparams.pop("one_minus_gae_lambda")

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
