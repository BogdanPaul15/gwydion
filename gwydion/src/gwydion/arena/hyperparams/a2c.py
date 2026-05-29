from typing import Any
import optuna

from .maps import ACTIVATION_FN_MAP, NET_ARCH_MAP

def sample_a2c_params(trial: optuna.Trial) -> dict:
    """Sample A2C hyperparameters for one Optuna trial."""
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 8) # 32 - 256, episode=25 so >=25 needed

    one_minus_gamma      = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    ent_coef      = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True) # floor raised to prevent premature determinism
    vf_coef       = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)
    rms_prop_eps  = trial.suggest_float("rms_prop_eps", 1e-6, 1e-3, log=True)
    use_rms_prop  = trial.suggest_categorical("use_rms_prop", [True, False])
    net_arch      = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("n_steps",    2 ** n_steps_pow)
    trial.set_user_attr("gamma",      1 - one_minus_gamma)
    trial.set_user_attr("gae_lambda", 1 - one_minus_gae_lambda)

    return {
        "n_steps_pow":           n_steps_pow,
        "one_minus_gamma":       one_minus_gamma,
        "one_minus_gae_lambda":  one_minus_gae_lambda,
        "learning_rate":         learning_rate,
        "ent_coef":              ent_coef,
        "vf_coef":               vf_coef,
        "max_grad_norm":         max_grad_norm,
        "rms_prop_eps":          rms_prop_eps,
        "use_rms_prop":          use_rms_prop,
        "net_arch":              net_arch,
        "activation_fn":         activation_fn,
    }

def convert_a2c_params(sampled: dict[str, Any], n_envs: int = 1) -> dict[str, Any]:
    """Translate raw sample_a2c_params() dict into A2C(**kwargs)."""
    del n_envs
    hyperparams = sampled.copy()

    hyperparams["n_steps"] = 2 ** hyperparams.pop("n_steps_pow")

    hyperparams["gamma"] = 1 - hyperparams.pop("one_minus_gamma")
    hyperparams["gae_lambda"] = 1 - hyperparams.pop("one_minus_gae_lambda")

    hyperparams["policy_kwargs"] = {
        "net_arch": NET_ARCH_MAP[hyperparams.pop("net_arch")],
        "activation_fn": ACTIVATION_FN_MAP[hyperparams.pop("activation_fn")],
    }

    return hyperparams
