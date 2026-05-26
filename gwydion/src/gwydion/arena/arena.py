import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import optuna
from optuna.samplers import TPESampler
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

from gwydion.envs import BaseEnv
from gwydion.rewards import RewardStrategy

from .registry import get_spec
from .utils import (
	EpisodeCallback,
	EpisodeMetricsWriter,
	OptunaPruningCallback,
	StepObsCallback,
	StepObsWriter,
)

logger = logging.getLogger(__name__)

class Arena:
	"""Central framework for tuning, training, testing and comparing RL agents.

	Attributes:
		env_cls: Gymnasium environment class (must subclass :class:`BaseEnv`).
		alg: Algorithm name as registered in the spec registry.
		spec: Resolved :class:`AlgorithmSpec`.
		n_envs: Number of parallel envs for tune and train. ``test()`` always
			uses a single env.
		seed: The random seed for reproducibility. Defaults to 42.
		experiment_label: Human-readable string identifying this experiment
			(e.g., ``"{AlgName}_{EnvName}"``)
		output_dir: Root directory; one subdirectory per ``experiment_label``.
	"""

	def __init__(self, env_cls: Type[BaseEnv], alg: str = "ppo",
				 n_envs: int = 1, seed: int = 42,
				 output_dir: str = "arena_results"):
		self.env_cls = env_cls
		self.alg = alg.lower()
		self.spec = get_spec(self.alg)
		self.n_envs = n_envs
		self.seed = seed

		self.experiment_label = f"{self.spec.cls.__name__}_{env_cls.__name__}"
		self.output_dir = Path(output_dir) / self.experiment_label
		self.output_dir.mkdir(parents=True, exist_ok=True)

	def tune(self, config_path: str, reward_strategy: RewardStrategy,
			 n_trials: int = 20, tune_steps: int = 100_000,
			 n_eval_episodes: int = 10, n_jobs: int = 1,
			 pruner: Optional[optuna.pruners.BasePruner] = None,
			 direction: str = "maximize") -> dict:
		"""Executes hyperparameter optimization using Optuna to find the best model configuration.

		Trials train on a vectorized simulation env (``n_envs`` workers) and
		report intermediate eval rewards so the pruner can kill bad trials
		early. The best params dict is saved to ``best_params.json`` next to
		the study DB and returned.

		Args:
			config_path: Path to the cluster config file used for tuning.
			reward_strategy: The reward strategy instance to use during tuning.
			n_trials: The number of hyperparameter configurations to sample and evaluate.
				Defaults to 20.
			tune_steps: The number of environment timesteps to train the model during
				each trial. Defaults to 100,000.
			n_eval_episodes: The number of episodes to run during intermediate pruning
				evaluations and the final trial evaluation. Defaults to 10.
			n_jobs: The number of parallel jobs to run for the Optuna study. Defaults to 1.
			pruner: The Optuna pruning strategy to use.
				If None, defaults to MedianPruner with 5 warmup steps.
			direction: The optimization direction for the study. Should be "maximize"
				(usually for reward) or "minimize". Defaults to "maximize".

		Returns:
			dict: A dictionary containing the best sampled hyperparameter values found
			during the study.
		"""
		pruner = pruner or optuna.pruners.MedianPruner(n_warmup_steps=5)

		tune_dir = self.phase_dir("tune", reward_strategy,
								   trials=n_trials, steps=tune_steps)

		study = optuna.create_study(
			storage=f"sqlite:///{tune_dir / 'optuna_study.db'}",
			study_name=f"{self.experiment_label}_{reward_strategy.__class__.__name__}",
			direction=direction,
			sampler=TPESampler(seed=self.seed, multivariate=True, n_startup_trials=5),
			pruner=pruner,
			load_if_exists=True,
		)

		def objective(trial: optuna.Trial) -> float:
			sampled = self.spec.sampler(trial)
			converted = self.spec.converter(sampled, self.n_envs)
			gamma = converted.get("gamma", 0.99)

			env = self.make_env(config_path, reward_strategy,
								 training=True, gamma=gamma)
			eval_env = self.make_env(config_path, reward_strategy,
									  training=False, gamma=gamma,
									  n_envs_override=1)

			try:
				model = self.build_model(env, converted)
				pruning_cb = OptunaPruningCallback(
					trial, eval_env,
					n_eval_episodes=n_eval_episodes,
					eval_freq=max(1, tune_steps // 5),
					maskable=self.spec.maskable,
				)
				model.learn(total_timesteps=tune_steps, callback=pruning_cb)

				eval_env.obs_rms = env.obs_rms
				eval_env.ret_rms = env.ret_rms
				evaluator = maskable_evaluate_policy if self.spec.maskable else evaluate_policy
				mean_reward, _ = evaluator(model, eval_env, n_eval_episodes=n_eval_episodes)
				return float(mean_reward)
			except optuna.TrialPruned:
				raise
			except (ValueError, RuntimeError) as e:
				logger.exception("[Trial %d] failed with %s — pruning", trial.number, type(e).__name__)
				raise optuna.TrialPruned() from e
			finally:
				env.close()
				eval_env.close()

		study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

		params_path = tune_dir / "best_params.json"
		params_path.write_text(
			json.dumps({
				"best_value": study.best_value,
				"best_params": study.best_params,
				"reward_strategy": reward_strategy.__class__.__name__,
			}, indent=2),
			encoding="utf-8")
		logger.info("Best params saved to %s", params_path)
		return study.best_params

	def train(self, config_path: str, reward_strategy: RewardStrategy,
			  hyperparams: Optional[Dict[str, Any]] = None,
			  total_steps: int = 500_000, save_freq: int = 50_000,
			  run_label: Optional[str] = None,
			  resume_from: Optional[tuple] = None,
			  extra_callbacks: Optional[list] = None,
			  record_step_obs: bool = False,
			  obs_feature_names: Optional[List[str]] = None):
		"""Trains the final model and persists artifacts + per-episode metrics.

		Returns a ``(model_path, stats_path)`` tuple so the caller can pass
		them to :meth:`test`. Writes ``episodes.csv`` and ``summary.json``
		into the run directory via :class:`EpisodeMetricsWriter`.
		"""
		converted = self.spec.converter(hyperparams, self.n_envs) if hyperparams else {}
		gamma = converted.get("gamma", 0.99)

		run_label = run_label or "default"
		run_dir = self.phase_dir("train", reward_strategy,
								  steps=total_steps, label=run_label)

		env = self.make_env(config_path, reward_strategy, training=True, gamma=gamma)

		if resume_from:
			model_path, stats_path = resume_from
			env = VecNormalize.load(stats_path, env.venv)
			env.training = True
			env.norm_reward = True
			model = self.spec.cls.load(model_path, env=env)
			reset_timesteps = False
		else:
			model = self.build_model(env, converted, tensorboard_log=str(run_dir / "tb"))
			reset_timesteps = True

		checkpoint_cb = CheckpointCallback(
			save_freq=max(save_freq // self.n_envs, 1),
			save_path=str(run_dir / "checkpoints"),
			name_prefix="model",
		)
		writer = EpisodeMetricsWriter(
			out_path=run_dir / "episodes.csv",
			phase="train",
			run_id=f"{self.alg}_{run_label}_{self.seed}",
		)
		callbacks = [checkpoint_cb, EpisodeCallback(writer)]
		if record_step_obs:
			obs_writer = StepObsWriter(
				out_path=run_dir / "step_obs.csv",
				feature_names=obs_feature_names or env.get_attr("obs_feature_names")[0],
			)
			callbacks.append(StepObsCallback(obs_writer))
		if extra_callbacks:
			callbacks.extend(extra_callbacks)

		model.learn(
			total_timesteps=total_steps,
			callback=callbacks,
			reset_num_timesteps=reset_timesteps,
		)

		model_path = run_dir / "model_final"
		stats_path = run_dir / "vecnormalize.pkl"
		model.save(str(model_path))
		env.save(str(stats_path))
		env.close()

		logger.info("Train complete — model %s | stats %s", model_path, stats_path)
		return str(model_path), str(stats_path)

	def test(self, config_path: str, reward_strategy: RewardStrategy,
			 model_path: str, stats_path: str,
			 n_episodes: int = 100, run_label: Optional[str] = None,
			 deterministic: bool = True,
			 record_step_obs: bool = False,
			 obs_feature_names: Optional[List[str]] = None) -> dict:
		"""Runs the trained model on the cluster (or any per-phase config).

		Records each episode to ``episodes.csv`` with ``phase="test"``
		and returns the summary dict.
		"""
		run_label = run_label or "default"
		run_dir = self.phase_dir("test", reward_strategy,
								  ep=n_episodes, label=run_label)

		env = self.make_env(config_path, reward_strategy,
							 training=False, gamma=0.99, n_envs_override=1)
		env = VecNormalize.load(stats_path, env.venv)
		env.training = False
		env.norm_reward = False

		model = self.spec.cls.load(model_path, env=env)

		writer = EpisodeMetricsWriter(
			out_path=run_dir / "episodes.csv",
			phase="test",
			run_id=f"{self.alg}_{run_label}_{self.seed}",
		)

		obs_writer = StepObsWriter(
			out_path=run_dir / "step_obs.csv",
			feature_names=obs_feature_names or env.get_attr("obs_feature_names")[0],
		) if record_step_obs else None

		completed = 0
		step = 0
		obs = env.reset()
		while completed < n_episodes:
			action, _ = model.predict(obs, deterministic=deterministic)
			obs, _, _, infos = env.step(action)
			step += 1
			raw_obs_list = env.get_attr("last_obs") if obs_writer else None
			for env_rank, info in enumerate(infos):
				if obs_writer and raw_obs_list[env_rank] is not None:
					obs_writer.record(
						step=step, env_rank=env_rank,
						obs=raw_obs_list[env_rank],
						latency=float(info.get("latency", 0.0)),
					)
				if "episode" in info:
					writer.record(info, env_rank=env_rank)
					completed += 1
					logger.info("Test episode %d/%d | reward=%.3f",
								completed, n_episodes,
								info["episode"].get("r", 0.0))
					if completed >= n_episodes:
						break

		if obs_writer:
			obs_writer.flush()
		env.close()
		summary = writer.write_summary()
		logger.info("Test complete — summary %s", run_dir / "summary.json")
		return summary

	def make_env(self, config_path: str, reward_strategy: RewardStrategy,
				  training: bool = True, gamma: float = 0.99,
				  n_envs_override: Optional[int] = None) -> VecNormalize:
		"""Builds 1 or multiple envs wrapped in Monitor + VecNormalize.

		Eval/test envs get a disjoint seed offset so they don't overlap with training.
		"""
		n = n_envs_override if n_envs_override is not None else self.n_envs
		base_seed = self.seed if training else self.seed + 10_000

		def _factory(rank: int):
			def _init():
				env = self.env_cls(
					config_path=config_path,
					reward_strategy=reward_strategy,
					seed=base_seed + rank,
				)
				return Monitor(env)
			return _init

		vec_cls = DummyVecEnv if n == 1 else SubprocVecEnv
		vec_env = vec_cls([_factory(i) for i in range(n)])
		return VecNormalize(vec_env, training=training,
							norm_reward=training, gamma=gamma)

	def build_model(self, env, hyperparams: Dict[str, Any],
					 tensorboard_log: Optional[str] = None) -> BaseAlgorithm:
		kwargs: Dict[str, Any] = {
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

	def phase_dir(self, phase: str, reward_strategy: RewardStrategy, **tags) -> Path:
		"""Builds and creates a per-phase output directory tagged with the
		reward strategy, the relevant phase tags, and a timestamp.

		Tags are formatted as ``key=value`` and dot-separated:
		e.g., ``phase_dir("train", CostStrategy(), steps=200, label="default")``
		yields ``train_CostStrategy_steps=200.label=default_<ts>``.
		"""
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		parts = [phase, reward_strategy.__class__.__name__]
		tag_parts = [f"{k}={v}" for k, v in tags.items() if v is not None]
		if tag_parts:
			parts.append(".".join(tag_parts))
		parts.append(ts)
		path = self.output_dir / "_".join(parts)
		path.mkdir(parents=True, exist_ok=True)
		return path
