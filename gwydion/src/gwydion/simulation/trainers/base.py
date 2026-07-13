from abc import ABC, abstractmethod

from pathlib import Path
from typing import List, Optional, Tuple
import logging
import math

import yaml
import numpy as np
import pandas as pd

from gwydion.simulation.models import SimulatorModel
from .utils import build_transitions, delta_columns

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
	"""Abstract base class for offline transition-model trainers.

	A trainer reads a YAML config describing the deployments and the per-model
	training settings, loads the observation dataset, performs a chronological
	train/val/test split, and exposes a uniform ``tune -> train -> test -> save``
	workflow. The fitted result is exported as a
	:class:`~gwydion.simulation.models.SimulatorModel` consumed at runtime by the
	learned simulation strategy.

	Attributes:
		model_key (str): The type of the model.
		df (pd.DataFrame): The full observation dataset.
		train_df / val_df / test_df (pd.DataFrame): Chronological splits.
		best_params (Optional[dict]): Best hyperparameters found by :meth:`tune`.
	"""

	model_key: str = "base"

	def __init__(self, config_path: str, seed: int = 42) -> None:
		"""Loads the config and dataset and builds the chronological splits.

		Args:
			config_path (str): Path to the trainer YAML config.
			seed (int): Random seed for reproducibility. Defaults to 42.
		"""
		self._cfg = self._load_config(config_path)
		self._training_cfg = self._cfg.get("training", {})
		self.seed = seed

		self.best_params: Optional[dict] = None

		self.df = self._load_dataset(self.dataset_path)
		self.train_df, self.val_df, self.test_df = self._split(self.df)

		logger.info("Trainer %s | dataset rows: %d | split: %d / %d / %d",
					self.model_key, len(self.df), len(self.train_df),
					len(self.val_df), len(self.test_df))

	@property
	def deployment_names(self) -> List[str]:
		"""Ordered deployment names declared in the config."""
		return [deployment["name"] for deployment in self._cfg.get("deployments", [])]

	@property
	def state_features(self) -> List[str]:
		"""Flattened ``{deployment}_{feature}`` state (input) column names."""
		return [f"{d['name']}_{feature}"
				for d in self._cfg["deployments"] for feature in d["state_features"]]

	@property
	def target_features(self) -> List[str]:
		"""Flattened ``{deployment}_{feature}`` target (output) column names."""
		return [f"{d['name']}_{feature}"
				for d in self._cfg["deployments"] for feature in d["target_features"]]

	@property
	def dataset_path(self) -> str:
		"""Path to the observation CSV as declared in the trainer config."""
		path = self._training_cfg.get("dataset")
		if not path:
			raise ValueError("No dataset configured. Set 'training.dataset' in the config.")
		return path

	@property
	def model_params(self) -> dict:
		"""The ``training.models.<model_key>`` sub-config for this trainer."""
		return self._training_cfg.get("models", {}).get(self.model_key, {})

	def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""Splits the dataset chronologically into train/val/test.

		The split is strictly time-ordered so that no future
		observation leaks into the training window.

		Args:
			df (pd.DataFrame): The full observation dataframe.

		Returns:
			Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test.
		"""
		split = self._training_cfg.get("split", {})
		train_frac = float(split.get("train", 0.70))
		val_frac = float(split.get("val", 0.15))

		ordered = df.sort_values("date").reset_index(drop=True)
		n = len(ordered)
		i_train = int(n * train_frac)
		i_val = int(n * (train_frac + val_frac))

		return (ordered.iloc[:i_train].copy(),
				ordered.iloc[i_train:i_val].copy(),
				ordered.iloc[i_val:].copy())

	@abstractmethod
	def tune(self, n_trials: int = 50) -> None:
		"""Searches hyperparameters with Optuna, storing them in ``best_params``."""

	@abstractmethod
	def train(self) -> None:
		"""Fits the final model on the training data (uses ``best_params`` if set)."""

	@abstractmethod
	def test(self) -> dict:
		"""Evaluates the trained model on the test split.

		Returns:
			dict: One-step per-target and aggregate error metrics.
		"""

	@abstractmethod
	def to_model(self) -> SimulatorModel:
		"""Exports the trained model as a runtime :class:`SimulatorModel`."""

	def rollout(self, n_steps: int = None, model: Optional[SimulatorModel] = None) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
		"""Autoregressive rollout on the test split using real action deltas.

		Seeds from the last ``required_history`` real rows before the test
		boundary, then chains ``model.predict_next`` calls, substituting
		predicted targets back into the state vector at each step while keeping
		real pod counts from the dataset.

		Args:
			n_steps: Rollout length. Defaults to the full test split length.

		Returns:
			Tuple: ``(y_pred, y_true, dates)`` — each of shape ``(N, n_targets)``
				plus a DatetimeIndex of length ``N``.
		"""
		if model is None:
			model = self.to_model()
		history_len = model.required_history

		transitions = build_transitions(self.deployment_names, self.df)
		states  = transitions[self.state_features].to_numpy(dtype=np.float64)
		deltas  = transitions[delta_columns(self.deployment_names)].to_numpy(dtype=np.float64)
		targets = transitions[self.target_features].to_numpy(dtype=np.float64)
		trans_dates = pd.to_datetime(transitions["date"])

		test_start = pd.to_datetime(self.test_df["date"].min())
		i_val = int((trans_dates >= test_start).idxmax())

		if n_steps is None:
			n_steps = len(transitions) - i_val - 1

		# Seed includes i_val itself so history[-1] is the state at the test boundary,
		# matching how the model sees "current state at t" during training.
		history = list(states[max(0, i_val - history_len + 1): i_val + 1])
		target_in_state = [self.state_features.index(f) for f in self.target_features]

		y_pred, y_true, pred_dates = [], [], []
		for step in range(n_steps):
			t = i_val + step
			if t + 1 >= len(transitions):
				break

			pred = model.predict_next(
				np.array(history[-history_len:], dtype=np.float64), deltas[t])

			y_pred.append(pred)
			y_true.append(targets[t + 1])
			pred_dates.append(trans_dates.iloc[t + 1])

			next_state = states[t + 1].copy()
			for pred_i, state_i in enumerate(target_in_state):
				next_state[state_i] = pred[pred_i]
			history.append(next_state)

		return (np.asarray(y_pred, dtype=np.float64),
				np.asarray(y_true, dtype=np.float64),
				pd.DatetimeIndex(pred_dates))

	def rollout_episodes(self, horizon: int = 25, n_rollouts: int = 200,
						 model: Optional[SimulatorModel] = None,
						 seed: Optional[int] = None) -> dict:
		"""Evaluates the model with many short autoregressive rollouts.

		This mirrors how the learned strategy is actually used: the environment
		reseeds the simulator from real data every ``max_steps`` steps, so the
		model only ever runs ``horizon`` autoregressive steps before getting
		fresh ground truth. Each rollout is seeded from a random real slice of
		the test split, rolled out ``horizon`` steps (real pod deltas, predicted
		targets fed back), and the per-step error is accumulated. The continuous
		full-split :meth:`rollout` is an unrealistically harsh worst case by
		comparison.

		Args:
			horizon: Autoregressive steps per rollout (match the env ``max_steps``).
			n_rollouts: Number of random start points sampled from the test split.
			model: Model to evaluate (defaults to ``self.to_model()``).
			seed: RNG seed for the start points (defaults to ``self.seed``).

		Returns:
			dict: ``mae``/``rmse``/``nrmse`` arrays of shape ``(horizon, n_targets)``
				(error at each rollout step per target), plus ``target_features``,
				``target_scale`` (per-target std used for the NRMSE), ``horizon``
				and the realised ``n_rollouts``.
		"""
		if model is None:
			model = self.to_model()
		history_len = model.required_history

		transitions = build_transitions(self.deployment_names, self.df)
		states  = transitions[self.state_features].to_numpy(dtype=np.float64)
		deltas  = transitions[delta_columns(self.deployment_names)].to_numpy(dtype=np.float64)
		targets = transitions[self.target_features].to_numpy(dtype=np.float64)
		trans_dates = pd.to_datetime(transitions["date"])

		test_start = pd.to_datetime(self.test_df["date"].min())
		i_val = int((trans_dates >= test_start).idxmax())

		target_in_state = [self.state_features.index(f) for f in self.target_features]
		n_targets = len(self.target_features)

		lo = max(i_val, history_len - 1)
		hi = len(transitions) - horizon - 1
		if hi <= lo:
			raise ValueError(
				f"Test split too short for horizon={horizon} "
				f"(usable starts: {hi - lo}).")

		rng = np.random.default_rng(self.seed if seed is None else seed)
		n_rollouts = int(min(n_rollouts, hi - lo))
		starts = rng.choice(np.arange(lo, hi), size=n_rollouts, replace=False)

		abs_err = np.zeros((horizon, n_targets))
		sq_err  = np.zeros((horizon, n_targets))
		for s in starts:
			history = [states[k].copy() for k in range(s - history_len + 1, s + 1)]
			for h in range(horizon):
				t = s + h
				pred = model.predict_next(
					np.array(history[-history_len:], dtype=np.float64), deltas[t])
				true = targets[t + 1]
				abs_err[h] += np.abs(pred - true)
				sq_err[h]  += (pred - true) ** 2

				next_state = states[t + 1].copy()
				for pred_i, state_i in enumerate(target_in_state):
					next_state[state_i] = pred[pred_i]
				history.append(next_state)

		abs_err /= n_rollouts
		rmse = np.sqrt(sq_err / n_rollouts)
		scale = targets.std(axis=0)
		scale = np.where(scale > 0, scale, 1.0)

		return {
			"horizon": horizon,
			"n_rollouts": n_rollouts,
			"mae": abs_err,
			"rmse": rmse,
			"nrmse": rmse / scale,
			"target_features": list(self.target_features),
			"target_scale": scale,
		}

	def save(self, path: Path) -> None:
		"""Exports and persists the trained model as an artifact directory.

		Args:
			path (Path): Destination artifact directory.
		"""
		model = self.to_model()
		model.save(Path(path))
		logger.info("Saved %s model artifact to %s", self.model_key, path)

	@staticmethod
	def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
						   target_features: List[str]) -> dict:
		"""Computes per-target and aggregate MAE/MSE/RMSE.

		Args:
			y_true (np.ndarray): Ground-truth targets, shape ``(N, n_target)``.
			y_pred (np.ndarray): Predicted targets, shape ``(N, n_target)``.
			target_features (List[str]): Target column names (for the per-target keys).

		Returns:
			dict: ``{"per_target": {col: {"mae", "mse", "rmse"}}, "mae", "mse", "rmse"}``.
		"""
		y_true = np.asarray(y_true, dtype=np.float64)
		y_pred = np.asarray(y_pred, dtype=np.float64)
		abs_err = np.abs(y_true - y_pred)
		sq_err = (y_true - y_pred) ** 2

		per_target = {}
		for i, col in enumerate(target_features):
			mse_i = float(sq_err[:, i].mean())
			per_target[col] = {
				"mae": float(abs_err[:, i].mean()),
				"mse": mse_i,
				"rmse": float(math.sqrt(mse_i)),
			}

		agg_mse = float(sq_err.mean())
		return {
			"per_target": per_target,
			"mae": float(abs_err.mean()),
			"mse": agg_mse,
			"rmse": float(math.sqrt(agg_mse)),
		}

	@staticmethod
	def _load_config(config_path: str) -> dict:
		"""Loads and validates a trainer YAML config.

		Args:
			config_path (str): Path to the YAML config file.

		Returns:
			dict: The parsed configuration.

		Raises:
			FileNotFoundError: If the config file does not exist.
			ValueError: If no deployments are declared.
		"""
		path = Path(config_path)
		if not path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")
		with open(path, encoding="utf-8") as f:
			cfg = yaml.safe_load(f)
		if not cfg.get("deployments"):
			raise ValueError("Config must define at least one deployment.")
		return cfg

	@staticmethod
	def _load_dataset(csv_path: str) -> pd.DataFrame:
		"""Loads the observation CSV, parsing and sorting by ``date``.

		Args:
			csv_path (str): Path to the observation CSV.

		Returns:
			pd.DataFrame: The chronologically sorted dataset.

		Raises:
			FileNotFoundError: If the CSV does not exist.
		"""
		if not Path(csv_path).exists():
			raise FileNotFoundError(f"Dataset not found: {csv_path}")
		df = pd.read_csv(csv_path, parse_dates=["date"])
		return df.sort_values("date").reset_index(drop=True)
