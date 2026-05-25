from pathlib import Path
import json

from .base import SimulatorModel
from .lgbm_model import LGBMSimulatorModel
from .lstm_model import LSTMSimulatorModel, TransitionLSTM
from .varima_model import VARIMASimulatorModel

_MODEL_TYPES = {
	cls.model_type: cls
	for cls in (LGBMSimulatorModel, LSTMSimulatorModel, VARIMASimulatorModel)
}

def load_simulator_model(path) -> SimulatorModel:
	"""Loads a simulator model artifact, dispatching on its recorded type.

	Args:
		path: Path to the artifact directory containing ``meta.json``.

	Returns:
		SimulatorModel: The reconstructed model instance.

	Raises:
		FileNotFoundError: If the directory or its ``meta.json`` is missing.
		ValueError: If ``meta.json`` declares an unknown ``model_type``.
	"""
	path = Path(path)
	meta_path = path / "meta.json"
	if not meta_path.exists():
		raise FileNotFoundError(f"No model artifact found at {path}.")

	model_type = json.loads(meta_path.read_text(encoding="utf-8")).get("model_type")
	cls = _MODEL_TYPES.get(model_type)
	if cls is None:
		raise ValueError(
			f"Unknown model_type '{model_type}' in {meta_path}. "
			f"Known types: {sorted(_MODEL_TYPES)}"
		)

	return cls.load(path)

__all__ = [
	"SimulatorModel",
	"LGBMSimulatorModel",
	"LSTMSimulatorModel",
	"TransitionLSTM",
	"VARIMASimulatorModel",
	"load_simulator_model",
]
