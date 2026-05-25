from .base import BaseTrainer
from .lgbm import LGBMTrainer
from .lstm import LSTMTrainer
from .varima import VARIMATrainer
from .utils import build_transitions

TRAINERS = {
	"lgbm": LGBMTrainer,
	"lstm": LSTMTrainer,
	"varima": VARIMATrainer,
}

def build_trainer(model: str, config_path: str) -> BaseTrainer:
	"""Instantiates a trainer by model name.

	Args:
		model (str): One of ``lgbm``, ``lstm`` or ``varima``.
		config_path (str): Path to the trainer YAML config.

	Returns:
		BaseTrainer: The instantiated trainer.

	Raises:
		ValueError: If ``model`` is not a known trainer.
	"""
	cls = TRAINERS.get(model)
	if cls is None:
		raise ValueError(f"Unknown model '{model}'. Known: {sorted(TRAINERS)}")
	return cls(config_path)

__all__ = ["BaseTrainer", "build_trainer", "build_transitions", "TRAINERS",
		   "LGBMTrainer", "LSTMTrainer", "VARIMATrainer"]
