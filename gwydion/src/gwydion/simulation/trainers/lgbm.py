from typing import Optional
import logging
import warnings

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from gwydion.simulation.models import LGBMSimulatorModel
from .base import BaseTrainer
from .utils import build_tabular_dataset, build_transitions
from gwydion.simulation.utils import add_temporal_columns, temporal_feature_names

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

DEFAULT_PARAMS = {
	"n_estimators": 400,
	"learning_rate": 0.05,
	"num_leaves": 63,
	"max_depth": 8,
	"min_child_samples": 30,
	"subsample": 0.9,
	"colsample_bytree": 0.9,
	"reg_lambda": 0.1,
}

class LGBMTrainer(BaseTrainer):
	"""Trains a one-step transition model with LightGBM gradient boosting.

	The transition is framed as tabular multi-output regression: the input row
	is the current state concatenated with the per-deployment pod delta and temporal features,
	and the output is the next-step target metrics. Each target is handled by its own
	LightGBM regressor via :class:`~sklearn.multioutput.MultiOutputRegressor`.
	"""

	model_key = "lgbm"

	def __init__(self, config_path: str) -> None:
		"""Loads data and builds the dataset table for each split.

		Args:
			config_path (str): Path to the trainer YAML config.
		"""
		super().__init__(config_path)

		self._model: Optional[MultiOutputRegressor] = None

		args = (self.deployment_names, self.state_features, self.target_features)
		full_x, full_y = build_tabular_dataset(self.df, *args)

		n_df = len(self.df)
		n = len(full_x)
		i_train = round(n * len(self.train_df) / n_df)
		i_val = round(n * (len(self.train_df) + len(self.val_df)) / n_df)

		self.x_train, self.y_train = full_x[:i_train], full_y[:i_train]
		self.x_val, self.y_val = full_x[i_train:i_val], full_y[i_train:i_val]
		self.x_test, self.y_test = full_x[i_val:], full_y[i_val:]

	def _build_regressor(self, params: dict) -> MultiOutputRegressor:
		"""Wraps a LightGBM regressor (with the given params) for multi-output use."""
		base = LGBMRegressor(verbose=-1, n_jobs=-1, subsample_freq=1, **params)
		return MultiOutputRegressor(base)

	def tune(self, n_trials: int = 50) -> None:
		study = optuna.create_study(direction="minimize")

		def objective(trial: optuna.Trial) -> float:
			params = {
				"n_estimators": trial.suggest_int("n_estimators", 100, 600),
				"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
				"num_leaves": trial.suggest_int("num_leaves", 15, 127),
				"max_depth": trial.suggest_int("max_depth", 3, 12),
				"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
				"subsample": trial.suggest_float("subsample", 0.6, 1.0),
				"colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
				"reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
			}
			model = self._build_regressor(params)
			model.fit(self.x_train, self.y_train)
			pred = model.predict(self.x_val)
			return float(np.sqrt(np.mean((self.y_val - pred) ** 2)))

		study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
		self.best_params = study.best_params
		logger.info("LGBM tuning done | best val RMSE: %.4f | params: %s",
					study.best_value, self.best_params)

	def train(self) -> None:
		params = dict(DEFAULT_PARAMS)
		params.update(self.model_params.get("defaults", {}))
		if self.best_params:
			params.update(self.best_params)

		x = np.concatenate([self.x_train, self.x_val], axis=0)
		y = np.concatenate([self.y_train, self.y_val], axis=0)

		self._model = self._build_regressor(params)
		self._model.fit(x, y)
		logger.info("LGBM trained on %d transitions with params: %s", len(x), params)

	def test(self) -> dict:
		if self._model is None:
			raise RuntimeError("Call train() before test().")
		pred = self._model.predict(self.x_test)
		return self.regression_metrics(self.y_test, pred, self.target_features)

	def predict_test(self):
		if self._model is None:
			raise RuntimeError("Call train() before predict_test().")

		transitions = build_transitions(self.deployment_names, self.df)
		transitions = add_temporal_columns(transitions, self.target_features)
		transitions = transitions.dropna(
			subset=temporal_feature_names(self.target_features)
		).reset_index(drop=True)

		i_val = len(self.x_train) + len(self.x_val)
		dates = pd.DatetimeIndex(transitions["date"].iloc[1:].values)[i_val:]
		pred = np.asarray(self._model.predict(self.x_test), dtype=np.float64)
		return pred, self.y_test, dates

	def to_model(self) -> LGBMSimulatorModel:
		if self._model is None:
			raise RuntimeError("Call train() before exporting the model.")

		return LGBMSimulatorModel(
			regressor=self._model,
			deployment_names=self.deployment_names,
			state_features=self.state_features,
			target_features=self.target_features,
			metadata={"trainer": "lgbm"},
		)
