from copy import deepcopy
from typing import Optional
import logging

import numpy as np
import pandas as pd
import optuna
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from gwydion.simulation.models import LSTMSimulatorModel, TransitionLSTM
from .base import BaseTrainer
from .utils import make_windows, build_transitions

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_PARAMS = {
	"hidden_size": 64,
	"num_layers": 1,
	"dropout": 0.1,
	"lr": 1e-3,
	"batch_size": 256,
	"epochs": 60,
	"tune_epochs": 15,
	"patience": 8,
}

class LSTMTrainer(BaseTrainer):
	"""Trains a sequence transition model with an action-conditioned LSTM.

	Sliding windows of recent states are encoded by an LSTM; the per-deployment
	pod delta is concatenated to the final hidden state before a regression head
	predicts the next-step target metrics. Inputs and outputs are standardized
	with scalers fitted on the training split only.
	"""

	model_key = "lstm"

	def __init__(self, config_path: str) -> None:
		"""Loads data, builds windows and fits the input/output scalers.

		Args:
			config_path (str): Path to the trainer YAML config.
		"""
		super().__init__(config_path)

		params = dict(DEFAULT_PARAMS)
		params.update(self.model_params.get("defaults", {}))
		self._defaults = params
		self.window = int(self.model_params.get("window", 16))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		torch.manual_seed(0)

		self._module: Optional[TransitionLSTM] = None
		self._hyperparams: Optional[dict] = None

		args = (self.deployment_names, self.state_features, self.target_features, self.window)
		seq_tr, act_tr, y_tr = make_windows(self.train_df, *args)
		seq_va, act_va, y_va = make_windows(self.val_df, *args)
		seq_te, act_te, y_te = make_windows(self.test_df, *args)

		# Fit scalers on training windows only, then standardize every split.
		n_state = len(self.state_features)
		state_scaler = StandardScaler().fit(seq_tr.reshape(-1, n_state))
		action_scaler = StandardScaler().fit(act_tr)
		target_scaler = StandardScaler().fit(y_tr)
		self._scalers = {"state": state_scaler, "action": action_scaler, "target": target_scaler}

		self._train = self._to_tensors(seq_tr, act_tr, y_tr)
		self._val = self._to_tensors(seq_va, act_va, y_va)
		self._test = self._to_tensors(seq_te, act_te, y_te)
		self._y_test_raw = y_te

	def _to_tensors(self, seq: np.ndarray, act: np.ndarray, y: np.ndarray) -> tuple:
		"""Standardizes a split and moves it to device tensors."""
		s = self._scalers
		orig = seq.shape
		seq = s["state"].transform(seq.reshape(-1, orig[-1])).reshape(orig)
		act = s["action"].transform(act)
		y = s["target"].transform(y)
		return (torch.tensor(seq, dtype=torch.float32, device=self.device),
				torch.tensor(act, dtype=torch.float32, device=self.device),
				torch.tensor(y, dtype=torch.float32, device=self.device))

	def _build_module(self, hp: dict) -> TransitionLSTM:
		"""Instantiates an untrained network for the given hyperparameters."""
		return TransitionLSTM(
			n_state=len(self.state_features),
			n_action=len(self.deployment_names),
			n_target=len(self.target_features),
			hidden_size=hp["hidden_size"],
			num_layers=hp["num_layers"],
			dropout=hp["dropout"],
		).to(self.device)

	def _fit_module(self, module: TransitionLSTM, hp: dict, epochs: int) -> float:
		"""Trains a module with early stopping on the validation loss.

		Args:
			module (TransitionLSTM): The network to train (modified in place).
			hp (dict): Hyperparameters (``lr``, ``batch_size``, ``patience``).
			epochs (int): Maximum number of epochs.

		Returns:
			float: Best validation MSE reached (module restored to that state).
		"""
		opt = torch.optim.Adam(module.parameters(), lr=hp["lr"])
		loss_fn = nn.MSELoss()
		seq_tr, act_tr, y_tr = self._train
		seq_va, act_va, y_va = self._val
		batch = hp["batch_size"]

		best_val, best_state, bad = float("inf"), None, 0
		for epoch in range(epochs):
			module.train()
			perm = torch.randperm(len(seq_tr), device=self.device)
			for i in range(0, len(seq_tr), batch):
				idx = perm[i:i + batch]
				opt.zero_grad()
				pred = module(seq_tr[idx], act_tr[idx])
				loss = loss_fn(pred, y_tr[idx])
				loss.backward()
				opt.step()

			module.eval()
			with torch.no_grad():
				val_loss = loss_fn(module(seq_va, act_va), y_va).item()

			if val_loss < best_val - 1e-6:
				best_val, best_state, bad = val_loss, deepcopy(module.state_dict()), 0
			else:
				bad += 1
				if bad >= hp["patience"]:
					break

		if best_state is not None:
			module.load_state_dict(best_state)
		return best_val

	def tune(self, n_trials: int = 50) -> None:
		study = optuna.create_study(direction="minimize")

		def objective(trial: optuna.Trial) -> float:
			hp = {
				"hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
				"num_layers": trial.suggest_int("num_layers", 1, 2),
				"dropout": trial.suggest_float("dropout", 0.0, 0.4),
				"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
				"batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
				"patience": self._defaults["patience"],
			}
			module = self._build_module(hp)
			return self._fit_module(module, hp, self._defaults["tune_epochs"])

		study.optimize(objective, n_trials=n_trials)
		self.best_params = study.best_params
		logger.info("LSTM tuning done | best val MSE: %.4f | params: %s",
					study.best_value, self.best_params)

	def train(self) -> None:
		hp = dict(self._defaults)
		if self.best_params:
			hp.update(self.best_params)

		module = self._build_module(hp)
		best_val = self._fit_module(module, hp, hp["epochs"])

		self._module = module
		self._hyperparams = hp
		logger.info("LSTM trained | window: %d | best val MSE: %.4f | params: %s",
					self.window, best_val, hp)

	def test(self) -> dict:
		if self._module is None:
			raise RuntimeError("Call train() before test().")

		seq_te, act_te, _ = self._test
		self._module.eval()
		with torch.no_grad():
			scaled = self._module(seq_te, act_te).cpu().numpy()

		pred = self._scalers["target"].inverse_transform(scaled)
		return self.regression_metrics(self._y_test_raw, pred, self.target_features)

	def predict_test(self):
		if self._module is None:
			raise RuntimeError("Call train() before predict_test().")

		transitions = build_transitions(self.deployment_names, self.test_df)
		# y[i] from make_windows corresponds to transitions[window + i], so dates
		# start at index self.window within the transitions frame.
		dates = pd.DatetimeIndex(transitions["date"].iloc[self.window:].values)

		seq_te, act_te, _ = self._test
		self._module.eval()
		with torch.no_grad():
			scaled = self._module(seq_te, act_te).cpu().numpy()

		pred = self._scalers["target"].inverse_transform(scaled)
		dates = dates[: len(pred)]
		return pred, self._y_test_raw, dates

	def to_model(self) -> LSTMSimulatorModel:
		if self._module is None:
			raise RuntimeError("Call train() before exporting the model.")
		return LSTMSimulatorModel(
			module=deepcopy(self._module).to("cpu"),
			window=self.window,
			scalers=self._scalers,
			deployment_names=self.deployment_names,
			state_features=self.state_features,
			target_features=self.target_features,
			metadata={"trainer": "lstm"},
		)
