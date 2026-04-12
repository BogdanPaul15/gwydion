import logging

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

import json
import yaml
import optuna
from optuna.samplers import TPESampler
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

from gwydion.envs import BaseEnv
from .registry import get_spec

logger = logging.getLogger(__name__)

class _OptunaPruningCallback(BaseCallback):
    """Periodically evaluates the model during an Optuna trial and prunes
    underperforming trials early based on intermediate results.

    Args:
        trial (optuna.Trial): The active Optuna trial being evaluated.
        eval_env (VecNormalize): A VecNormalize-wrapper environment used for evaluation.
            Must have training=False so statistics are not updated during eval.
        n_eval_episodes (int): Number of episodes to run per evaluation.
        eval_freq (int): Evaluate every eval_freq training steps.
        maskable (bool): If True, uses MaskablePPO evaluation.
    """
    def __init__(self, trial, eval_env, n_eval_episodes=5, eval_freq=10_000, maskable=False):
        """Initializes the callback with the trial and evaluation parameters."""
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.maskable = maskable

    def _on_step(self) -> bool:
        """Evaluates the policy at set intervals and triggers pruning if needed.
        
        Returns:
            bool: Always returns True. If it returned False, training would abort early.
        """
        if self.n_calls % self.eval_freq == 0:
            # The eval env needs the same normalization statistics as the training env,
            # otherwise the model receives observations on a different scale than what
            # it was trained on. The eval env keeps training=False so these stats are
            # used for normalization but never updated during evaluation.
            train_env = self.model.get_env()
            self.eval_env.obs_rms = train_env.obs_rms
            self.eval_env.ret_rms = train_env.ret_rms

            if self.maskable:
                mean_reward, _ = maskable_evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            else:
                mean_reward, _ = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)

            # Report intermediate result to Optuna so the pruner can compare
            # this trial against others at the same training step.
            self.trial.report(mean_reward, step=self.n_calls)

            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True

class _EpisodeCallback(BaseCallback):
    """A custom callback for collecting episode statistics in memory during training.

    This callback intercepts the `info` dictionaries returned by the environment 
    at each step. When an episode terminates, it extracts the standard episode 
    metrics (reward, length, time) computed by the `Monitor` wrapper, alongside 
    custom domain-specific metrics (average pods, average latency). 

    By storing these in memory rather than writing to disk on every episode, 
    it minimizes I/O overhead during fast rollouts.

    Attributes:
        episode_rewards (list[float]): A record of the unnormalized total reward for each completed episode.
        episode_lengths (list[int]): A record of the total number of steps taken in each completed episode.
        episode_times (list[float]): A record of the elapsed time (in seconds) for each completed episode.
        avg_pods (list[float]): A custom metric tracking the average number of active pods per episode.
        avg_latency (list[float]): A custom metric tracking the average system latency per episode.
    """
    def __init__(self):
        """Initializes the callback and the empty tracking arrays."""
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.avg_pods: list[float] = []
        self.avg_latency: list[float] = []

    def _on_step(self) -> bool:
        """Executes on every environment step to parse the vectorized `infos` dictionaries.

        It looks for the `"episode"` key, which is automatically injected by the 
        Gymnasium `Monitor` wrapper exactly on the step an environment resets.

        Returns:
            bool: Always returns True. If it returned False, training would abort early.
        """
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_times.append(float(info["episode"]["t"]))
                self.avg_pods.append(float(info.get("avg_pods", 0)))
                self.avg_latency.append(float(info.get("avg_latency", 0)))

        return True

class Arena:
    """Central framework for tuning, training, testing and comparing RL agents.
    
    Attributes:
        env_cls (Type[BaseEnv]): Gymnasium environment class (must subclass BaseEnv).
        env_kwargs (Dict[str, Any]): Keyword arguments forwarded to env_cls on every instantiation.
        n_envs (int, optional): The number of parallel environments to create. Defaults to 1.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        spec (Type[AlgorithmSpec]): Algorithm specification (class, policy, sampler, converter) 
            resolved from the algorithm name via the registry.
        experiment_label: Human-readable string identifying this experiment
            (e.g., "MaskablePPO_Redis_CostStrategy_sim").
        output_dir (str): Root directory for all experiment artifacts.
    """
    def __init__(self, env_cls: Type[BaseEnv], env_kwargs: Dict[str, Any],
                 alg: str = "ppo", n_envs: int = 1, seed: int = 42,
                 output_dir: str = "arena_results") -> None:
        """Initializes the Arena and creates the experiment output directory.
        
        Args:
            env_cls (Type[BaseEnv]): Gymnasium environment class (must subclass BaseEnv).
            env_kwargs (Dict[str, Any]): Keyword arguments passed to the environment constructor.
                Must include "reward_strategy" (a RewardStrategy instance) and
                "config_path" (path to the environment YAML config).
            alg (str): Algorithm name as registered in the spec registry
                (e.g., "ppo", "maskable_ppo").
            n_envs (int, optional): Number of parallel environments for training. Higher values
                increase data collection throughput but consume more CPU cores.
                Evaluation always uses a single environment.
            seed (int, optional): Global random seed. Environments receive seed + rank offsets
                so parallel envs produce different trajectories.
            output_dir (str): Parent directory under which the experiment folder is created.
        """
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs
        self.n_envs = n_envs
        self.seed = seed

        alg = alg.lower()
        self.spec = get_spec(alg)

        reward_strategy = env_kwargs.get("reward_strategy").__class__.__name__
        cfg_path = env_kwargs.get("config_path", "")

        try:
            with open(cfg_path, encoding="utf-8") as f:
                k8s = yaml.safe_load(f).get("env", {}).get("k8s", False)
        except (FileNotFoundError, yaml.YAMLError, AttributeError):
            k8s = False

        mode = "k8s" if k8s else "sim"

        env_name = env_cls.__name__
        alg_name = self.spec.cls.__name__

        self.experiment_label = f"{alg_name}_{env_name}_{reward_strategy}_{mode}"
        self.output_dir = Path(output_dir) / self.experiment_label
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tune(self, n_trials: int = 20, tune_steps: int = 100_000, n_eval_episodes: int = 10,
             n_jobs: int = 1, pruner: Optional[optuna.pruners.BasePruner] = None,
             direction: str = "maximize"):
        """Executes hyperparameter optimization using Optuna to find the best model configuration.
        
        This method sets up an Optuna study backed by a SQLite database (allowing for run 
        resumption), samples hyperparameters, trains a model for each trial, and periodically 
        evaluates performance to prune unpromising trials. The best hyperparameters are 
        returned and automatically saved to disk.

        Args:
            n_trials (int): The number of hyperparameter configurations to sample and evaluate.
                Defaults to 20.
            tune_steps (int): The number of environment timesteps to train the model during 
                each trial. Defaults to 100,000.
            n_eval_episodes (int): The number of episodes to run during intermediate pruning 
                evaluations and the final trial evaluation. Defaults to 10.
            n_jobs (int): The number of parallel jobs to run for the Optuna study. Defaults to 1.
            pruner (Optional[optuna.pruners.BasePruner]): The Optuna pruning strategy to use. 
                If None, defaults to MedianPruner with 5 warmup steps.
            direction (str): The optimization direction for the study. Should be "maximize" 
                (usually for reward) or "minimize". Defaults to "maximize".

        Returns:
            dict: A dictionary containing the best sampled hyperparameter values found 
            during the study.

        Notes:
            - The study is saved to a SQLite database (`optuna_study.db`) in a dynamically 
              generated output directory, meaning interrupted tuning sessions can be resumed.
            - Intermediate performance is evaluated approximately 5 times per trial.
            - Model observation and reward normalization statistics are automatically 
              synchronized from the training environment to the evaluation environment 
              before the final assessment.
            - The best parameters are serialized and saved to `best_params.json` in the 
              tuning output directory.
        """
        pruner = pruner or optuna.pruners.MedianPruner(n_warmup_steps=5)

        tune_name = f"tune_{tune_steps}_steps_{n_trials}_trials_{pruner.__class__.__name__}"
        tune_dir = self.output_dir / tune_name
        tune_dir.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            storage=f"sqlite:///{tune_dir / 'optuna_study.db'}",
            study_name=self.experiment_label,
            direction=direction,
            sampler=TPESampler(seed=self.seed, multivariate=True, n_startup_trials=5),
            pruner=pruner,
            load_if_exists=True
        )

        def objective(trial: optuna.Trial) -> float:
            sampled = self.spec.sampler(trial, n_envs=self.n_envs)
            converted = self.spec.converter(sampled)

            gamma = converted.get("gamma", 0.99)
            env = self._make_env(training=True, gamma=gamma)
            eval_env = self._make_env(training=False, gamma=gamma)
            maskable = self.spec.cls.__name__ == "MaskablePPO"

            try:
                model = self._build_model(env, converted)

                pruning_callback = _OptunaPruningCallback(
                    trial, eval_env,
                    n_eval_episodes=n_eval_episodes,
                    eval_freq=max(1, tune_steps // 5),
                    maskable=maskable,
                )

                model.learn(total_timesteps=tune_steps, callback=pruning_callback)

                # Final eval with latest stats
                eval_env.obs_rms = env.obs_rms
                eval_env.ret_rms = env.ret_rms

                if maskable:
                    mean_reward, _ = maskable_evaluate_policy(model, eval_env,
                                                              n_eval_episodes=n_eval_episodes)
                else:
                    mean_reward, _ = evaluate_policy(model, eval_env,
                                                     n_eval_episodes=n_eval_episodes)

                return float(mean_reward)
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning("[Trial %d] failed — %s", trial.number, e)
                raise optuna.TrialPruned() from e
            finally:
                env.close()
                eval_env.close()

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        best_sampled = study.best_params

        params_path = tune_dir / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({"best_value": study.best_value, "best_params": best_sampled}, f, indent=2)
        logger.info("Best params saved to %s", params_path)

        return best_sampled

    def train(self,):
        """TODO"""

    def test(self):
        """TODO"""

    def _make_env(self, training: bool = True, gamma: float = 0.99):
        # TODO don't use same seed for evaluation
        def _factory(rank: int):
            def _init():
                env = Monitor(self.env_cls(**self.env_kwargs))
                env.reset(seed=self.seed + rank)
                return env
            return _init

        if self.n_envs == 1:
            vec_env = DummyVecEnv([_factory(0)])
        else:
            vec_env = SubprocVecEnv([_factory(i) for i in range(self.n_envs)])

        return VecNormalize(vec_env, training=training, norm_reward=training, gamma=gamma)

    def _build_model(self, env, hyperparams: Dict[str, Any],
                     tensorboard_log: Optional[str] = None) -> BaseAlgorithm:
        kwargs = {
            "policy": self.spec.policy,
            "env": env,
            "verbose": 0,
            "device": "cpu",
            "seed": self.seed,
            **hyperparams,
        }

        if tensorboard_log:
            kwargs["tensorboard_log"] = tensorboard_log

        return self.spec.cls(**kwargs)
