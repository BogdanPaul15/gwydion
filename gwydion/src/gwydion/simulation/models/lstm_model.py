from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
import torch
from torch import nn

from .base import SimulatorModel

class TransitionLSTM(nn.Module):
	"""Recurrent next-state predictor.

	A multi-layer LSTM encodes a window of recent cluster states; the final
	hidden state is concatenated with the (scaled) per-deployment pod delta and
	passed through an MLP head that regresses the next-step target metrics.
	"""

	def __init__(self, n_state: int, n_action: int, n_target: int,
				 hidden_size: int, num_layers: int, dropout: float) -> None:
		"""Builds the LSTM encoder and the action-conditioned regression head.

		Args:
			n_state (int): Number of state features per timestep.
			n_action (int): Number of action (pod-delta) features.
			n_target (int): Number of target metrics to predict.
			hidden_size (int): LSTM hidden dimension.
			num_layers (int): Number of stacked LSTM layers.
			dropout (float): Dropout applied between LSTM layers (>1 layer only).
		"""
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=n_state,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.head = nn.Sequential(
			nn.Linear(hidden_size + n_action, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_target),
		)

	def forward(self, sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Predicts scaled next-step targets for a batch of windows.

		Args:
			sequence (torch.Tensor): State windows of shape ``(B, W, n_state)``.
			action (torch.Tensor): Action features of shape ``(B, n_action)``.

		Returns:
			torch.Tensor: Predicted targets of shape ``(B, n_target)``.
		"""
		encoded, _ = self.lstm(sequence)
		last = encoded[:, -1, :]
		return self.head(torch.cat([last, action], dim=1))

class LSTMSimulatorModel(SimulatorModel):
	"""Sequence transition model wrapping a trained :class:`TransitionLSTM`.

	Requires a window of past states (``required_history == window``) and applies
	standardization to inputs/outputs using scalers fitted on the training split.
	"""

	model_type = "lstm"

	def __init__(self, module: TransitionLSTM, window: int, scalers: dict,
				 deployment_names: List[str], state_features: List[str],
				 target_features: List[str], metadata: Optional[dict] = None) -> None:
		"""Initializes the model around a trained module and its scalers.

		Args:
			module (TransitionLSTM): Trained network (set to eval mode here).
			window (int): Number of timesteps per input window.
			scalers (dict): ``StandardScaler`` objects keyed ``state``,
				``action``, ``target``.
			deployment_names (List[str]): Ordered deployment names.
			state_features (List[str]): Ordered state (input) column names.
			target_features (List[str]): Ordered target (output) column names.
			metadata (Optional[dict]): Free-form training metadata.
		"""
		super().__init__(deployment_names, state_features, target_features, metadata)
		self.module = module.eval()
		self.window = window
		self.scalers = scalers

	@property
	def required_history(self) -> int:
		return self.window

	def predict_next(self, history: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
		window = np.asarray(history, dtype=np.float64)[-self.window:]
		if len(window) < self.window:
			pad = np.repeat(window[:1], self.window - len(window), axis=0)
			window = np.concatenate([pad, window], axis=0)

		scaled_state = self.scalers["state"].transform(window)
		scaled_action = self.scalers["action"].transform(
			np.asarray(action_delta, dtype=np.float64).reshape(1, -1))

		with torch.no_grad():
			seq = torch.tensor(scaled_state, dtype=torch.float32).unsqueeze(0)
			act = torch.tensor(scaled_action, dtype=torch.float32)
			scaled_pred = self.module(seq, act).squeeze(0).numpy()

		return self.scalers["target"].inverse_transform(
			scaled_pred.reshape(1, -1))[0]

	def save(self, path: Path) -> None:
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)
		torch.save(self.module.state_dict(), path / "model.pt")
		joblib.dump(self.scalers, path / "scalers.joblib")

		meta = self._base_meta()
		meta["hyperparams"] = {
			"window": self.window,
			"hidden_size": self.module.lstm.hidden_size,
			"num_layers": self.module.lstm.num_layers,
			"dropout": float(self.module.lstm.dropout),
		}
		self._write_meta(path, meta)

	@classmethod
	def load(cls, path: Path) -> "LSTMSimulatorModel":
		path = Path(path)
		meta = cls._read_meta(path)
		hp = meta["hyperparams"]

		module = TransitionLSTM(
			n_state=len(meta["state_features"]),
			n_action=len(meta["deployment_names"]),
			n_target=len(meta["target_features"]),
			hidden_size=hp["hidden_size"],
			num_layers=hp["num_layers"],
			dropout=hp["dropout"],
		)
		module.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
		scalers = joblib.load(path / "scalers.joblib")

		return cls(module, hp["window"], scalers, meta["deployment_names"],
				   meta["state_features"], meta["target_features"], meta.get("metadata"))
