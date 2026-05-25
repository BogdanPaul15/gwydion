from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib

from .base import SimulatorModel

class VARIMASimulatorModel(SimulatorModel):
	"""Vector-autoregression transition model.

	The target metrics form the endogenous multivariate series and the
	per-deployment pod counts act as exogenous regressors — this is how the
	scaling action enters the model. A one-step forecast is produced from the
	last ``k_ar`` states and the post-action pod counts. Endogenous and
	exogenous series are standardized with scalers fitted on the training split.
	"""

	model_type = "varima"

	def __init__(self, results, scalers: dict, deployment_names: List[str],
				 state_features: List[str], target_features: List[str],
				 metadata: Optional[dict] = None) -> None:
		"""Initializes the model around a fitted VAR results object.

		Args:
			results: A fitted statsmodels ``VARResults`` instance.
			scalers (dict): ``StandardScaler`` objects keyed ``endog`` and ``exog``.
			deployment_names (List[str]): Ordered deployment names.
			state_features (List[str]): Ordered state (input) column names.
			target_features (List[str]): Ordered target (output) column names.
			metadata (Optional[dict]): Free-form training metadata.
		"""
		super().__init__(deployment_names, state_features, target_features, metadata)
		self.results = results
		self.k_ar = max(int(results.k_ar), 1)
		self.scalers = scalers

		# Column positions of the pod-count exogenous regressors in the state vector.
		self.pod_indices = [self.state_features.index(f"{n}_num_pods")
							  for n in self.deployment_names]

	@property
	def required_history(self) -> int:
		return self.k_ar

	def predict_next(self, history: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
		history = np.asarray(history, dtype=np.float64)[-self.k_ar:]

		endog = history[:, self.target_indices]
		next_exog = history[-1, self.pod_indices] + np.asarray(action_delta, dtype=np.float64)

		endog_scaled = self.scalers["endog"].transform(endog)
		next_exog_scaled = self.scalers["exog"].transform(next_exog.reshape(1, -1))

		forecast_scaled = np.asarray(self.results.forecast(endog_scaled, steps=1,
														   exog_future=next_exog_scaled))[0]
		return self.scalers["endog"].inverse_transform(forecast_scaled.reshape(1, -1))[0]

	def save(self, path: Path) -> None:
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)
		joblib.dump(self.results, path / "model.joblib")
		joblib.dump(self.scalers, path / "scalers.joblib")
		self._write_meta(path, self._base_meta())

	@classmethod
	def load(cls, path: Path) -> "VARIMASimulatorModel":
		path = Path(path)
		meta = cls._read_meta(path)
		results = joblib.load(path / "model.joblib")
		scalers = joblib.load(path / "scalers.joblib")

		return cls(results, scalers, meta["deployment_names"],
				   meta["state_features"], meta["target_features"], meta.get("metadata"))
