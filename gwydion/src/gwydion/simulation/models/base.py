from abc import ABC, abstractmethod

from pathlib import Path
from typing import List, Optional
import json

import numpy as np

class SimulatorModel(ABC):
	"""Runtime transition model used by the learned simulation strategy.

	A ``SimulatorModel`` predicts the next-step resource metrics of every
	deployment given a window of recent states and the per-deployment
	scaling action (pod-count delta). It is the inference artifact
	produced by a trainer, so the gym environment can load it without
	pulling in the training stack.

	Attributes:
		deployment_names (List[str]): Ordered deployment names.
		state_features (List[str]): Ordered ``{deployment}_{feature}`` columns
			describing a full cluster state (model input).
		target_features (List[str]): Ordered ``{deployment}_{feature}`` columns
			the model predicts (model output).
		metadata (dict): Free-form training metadata.
	"""

	model_type: str = "base"

	def __init__(self, deployment_names: List[str], state_features: List[str],
				 target_features: List[str], metadata: Optional[dict] = None) -> None:
		"""Initializes the simulator model.

		Args:
			deployment_names (List[str]): Ordered deployment names.
			state_features (List[str]): Ordered state (input) column names.
			target_features (List[str]): Ordered target (output) column names.
			metadata (Optional[dict]): Free-form training metadata.
		"""
		self.deployment_names = list(deployment_names)
		self.state_features = list(state_features)
		self.target_features = list(target_features)
		self.metadata = dict(metadata or {})

		# Position of each target column inside the state vector
		self.target_indices = [self.state_features.index(c) for c in self.target_features]

	@property
	def required_history(self) -> int:
		"""Number of past state vectors :meth:`predict_next` requires."""
		return 1

	@abstractmethod
	def predict_next(self, history: np.ndarray, action_delta: np.ndarray) -> np.ndarray:
		"""Predicts the next-step target metrics for one transition.

		Args:
			history (np.ndarray): Chronological state window of shape
				``(h, len(state_features))`` with ``h >= required_history``;
				``history[-1]`` is the most recent state.
			action_delta (np.ndarray): Per-deployment pod-count delta of shape
				``(len(deployment_names),)`` for the upcoming transition.

		Returns:
			np.ndarray: Predicted target vector of shape ``(len(target_features),)``.
		"""

	@abstractmethod
	def save(self, path: Path) -> None:
		"""Persists the model as a self-contained artifact directory.

		Args:
			path (Path): Destination directory.
		"""

	@classmethod
	@abstractmethod
	def load(cls, path: Path) -> "SimulatorModel":
		"""Loads a model of this exact type from an artifact directory."""

	def _base_meta(self) -> dict:
		"""Builds the ``meta.json`` payload shared by every artifact type."""
		return {
			"model_type": self.model_type,
			"deployment_names": self.deployment_names,
			"state_features": self.state_features,
			"target_features": self.target_features,
			"metadata": self.metadata,
		}

	@staticmethod
	def _write_meta(path: Path, meta: dict) -> None:
		"""Writes ``meta.json`` into the artifact directory."""
		(path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

	@staticmethod
	def _read_meta(path: Path) -> dict:
		"""Reads ``meta.json`` from the artifact directory."""
		return json.loads((path / "meta.json").read_text(encoding="utf-8"))
