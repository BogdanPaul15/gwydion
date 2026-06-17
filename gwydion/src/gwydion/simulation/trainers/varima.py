import logging
import warnings

import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

from gwydion.simulation.models import VARIMASimulatorModel
from .base import BaseTrainer
from .utils import make_endog_exog, build_transitions, delta_columns

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_PARAMS = {
	"p": 2,
	"trend": "c",
	"eval_stride": 5,
	"test_samples": 500,
}

class VARIMATrainer(BaseTrainer):
	"""Trains a vector-autoregression transition model with statsmodels VAR.

	The target metrics form the endogenous multivariate series and the
	per-deployment pod counts act as exogenous regressors — this is how the
	scaling action enters the model. Series are standardized with scalers
	fitted on the training split.
	"""

	model_key = "varima"

	def __init__(self, config_path: str, seed: int = 42) -> None:
		"""Loads data and builds the standardized endogenous/exogenous arrays.

		Args:
			config_path (str): Path to the trainer YAML config.
			seed (int): Random seed for reproducibility. Defaults to 42.
		"""
		super().__init__(config_path, seed=seed)

		params = dict(DEFAULT_PARAMS)
		params.update(self.model_params.get("defaults", {}))
		self._defaults = params

		self._results = None

		endog_tr, exog_tr = make_endog_exog(self.train_df, self.deployment_names,
											self.target_features)
		endog_va, exog_va = make_endog_exog(self.val_df, self.deployment_names,
											self.target_features)

		self._endog_scaler = StandardScaler().fit(endog_tr)
		self._exog_scaler = StandardScaler().fit(exog_tr)

		# Tiny jitter breaks exact collinearity between target columns (e.g. two
		# deployments sharing an identical latency series).
		jitter = np.random.default_rng(self.seed).normal(0.0, 1e-3, endog_tr.shape)
		self._endog_train = self._endog_scaler.transform(endog_tr) + jitter
		self._exog_train = self._exog_scaler.transform(exog_tr)
		self._endog_val = self._endog_scaler.transform(endog_va)
		self._exog_val = self._exog_scaler.transform(exog_va)

	def _fit_var(self, endog: np.ndarray, exog: np.ndarray, p: int, trend: str):
		"""Fits a VAR model of lag order ``p`` by ordinary least squares."""
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			return VAR(endog, exog=exog).fit(maxlags=p, trend=trend)

	def _one_step_rmse(self, results, endog: np.ndarray, exog: np.ndarray) -> float:
		"""Strided one-step-ahead forecast RMSE (in standardized units)."""
		k_ar = max(results.k_ar, 1)
		stride = self._defaults["eval_stride"]

		errors = []
		for t in range(k_ar, len(endog), stride):
			forecast = results.forecast(endog[t - k_ar:t], steps=1,
										exog_future=exog[t:t + 1])
			errors.append(endog[t] - forecast[0])

		return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")

	def tune(self, n_trials: int = 50) -> None:
		study = optuna.create_study(direction="minimize")

		def objective(trial: optuna.Trial) -> float:
			p = trial.suggest_int("p", 1, 4)
			trend = trial.suggest_categorical("trend", ["n", "c", "ct"])
			try:
				results = self._fit_var(self._endog_train, self._exog_train, p, trend)
				return self._one_step_rmse(results, self._endog_val, self._exog_val)
			except (ValueError, np.linalg.LinAlgError) as exc:
				logger.warning("VAR trial (p=%d,trend=%s) failed: %s", p, trend, exc)
				return float("inf")

		study.optimize(objective, n_trials=n_trials)
		self.best_params = study.best_params
		logger.info("VARIMA tuning done | best val RMSE: %.4f | params: %s",
					study.best_value, self.best_params)

	def train(self) -> None:
		params = dict(self._defaults)
		if self.best_params:
			params.update(self.best_params)

		# Fit on train + val (chronologically contiguous) for the final model.
		endog = np.concatenate([self._endog_train, self._endog_val], axis=0)
		exog = np.concatenate([self._exog_train, self._exog_val], axis=0)

		self._results = self._fit_var(endog, exog, params["p"], params["trend"])
		logger.info("VARIMA trained on %d observations | lag order p=%d trend=%s",
					len(endog), self._results.k_ar, params["trend"])

	def test(self) -> dict:
		if self._results is None:
			raise RuntimeError("Call train() before test().")

		model = self.to_model()
		transitions = build_transitions(self.deployment_names, self.test_df)
		states = transitions[self.state_features].to_numpy(dtype=np.float64)
		deltas = transitions[delta_columns(self.deployment_names)].to_numpy(dtype=np.float64)
		targets = transitions[self.target_features].to_numpy(dtype=np.float64)

		first = model.required_history - 1
		last = len(transitions) - 2
		if last < first:
			raise RuntimeError("Test split too small for the configured lag order.")
		n_samples = min(self._defaults["test_samples"], last - first + 1)
		indices = np.linspace(first, last, n_samples).astype(int)

		y_true, y_pred = [], []
		for t in indices:
			window = states[t - model.required_history + 1: t + 1]
			y_pred.append(model.predict_next(window, deltas[t]))
			y_true.append(targets[t + 1])

		return self.regression_metrics(np.asarray(y_true), np.asarray(y_pred),
									   self.target_features)

	def to_model(self) -> VARIMASimulatorModel:
		if self._results is None:
			raise RuntimeError("Call train() before exporting the model.")
		scalers = {"endog": self._endog_scaler, "exog": self._exog_scaler}
		return VARIMASimulatorModel(
			results=self._results,
			scalers=scalers,
			deployment_names=self.deployment_names,
			state_features=self.state_features,
			target_features=self.target_features,
			metadata={"trainer": "varima"},
		)
