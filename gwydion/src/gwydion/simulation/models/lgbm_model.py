from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import joblib

from .base import SimulatorModel
from gwydion.simulation.utils import (
	compute_temporal_features, temporal_max_lookback)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

class LGBMSimulatorModel(SimulatorModel):
	"""Gradient-boosted (LightGBM) one-step transition model.

	To compensate for LightGBM's lack of temporal awareness,
	the feature row at step ``t`` is formed of:
	``[state(t), temporal(t), pod_delta(t)]``,

	where ``temporal(t)`` contains lag/rolling stats.
	"""

	model_type = "lgbm"

	def __init__(self, regressor, deployment_names: List[str], state_features: List[str],
				 target_features: List[str], metadata: Optional[dict] = None) -> None:
		"""Initializes the model around a fitted regressor.

		Args:
			regressor: A fitted ``MultiOutputRegressor`` of LightGBM regressors.
			deployment_names (List[str]): Ordered deployment names.
			state_features (List[str]): Ordered state (input) column names.
			target_features (List[str]): Ordered target (output) column names.
			metadata (Optional[dict]): Free-form training metadata.
		"""
		super().__init__(deployment_names, state_features, target_features, metadata)
		self.regressor = regressor

	@property
	def required_history(self) -> int:
		# +1 for the current state itself; lookback covers the longest lag /
		# rolling window needed by the temporal feature block.
		return temporal_max_lookback() + 1

	def predict_next(self, history: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
		history = np.asarray(history, dtype=np.float64)
		state = history[-1]
		action = np.asarray(action_delta, dtype=np.float64)
		temporal = compute_temporal_features(history, self.target_indices)
		features = np.concatenate([state, temporal, action]).reshape(1, -1)
		return np.asarray(self.regressor.predict(features)[0], dtype=np.float64)

	def save(self, path: Path) -> None:
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)
		joblib.dump(self.regressor, path / "model.joblib")
		self._write_meta(path, self._base_meta())

	@classmethod
	def load(cls, path: Path) -> "LGBMSimulatorModel":
		path = Path(path)
		meta = cls._read_meta(path)
		regressor = joblib.load(path / "model.joblib")
		return cls(regressor, meta["deployment_names"], meta["state_features"],
				   meta["target_features"], meta.get("metadata"))
