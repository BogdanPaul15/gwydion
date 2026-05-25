import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import build_trainer

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"

def plot_rollout(trainer, y_pred: np.ndarray, y_true: np.ndarray,
				  dates: pd.DatetimeIndex, out_dir: Path,
				  rolling_window: int = None) -> list:
	"""Saves one PNG per deployment showing train+val context and rollout predictions.

	Each figure has one subplot per target feature. The grey region is the
	train+val actual series; the blue/orange lines are test actual/rollout predicted.
	Optionally applies a rolling moving average to the test visualizations.

	Args:
		trainer: A fitted :class:`BaseTrainer` instance.
		y_pred: Rollout predictions, shape ``(N, n_targets)``.
		y_true: Ground-truth values, shape ``(N, n_targets)``.
		dates: DatetimeIndex of length ``N`` — timestamps of predicted values.
		out_dir: Directory where PNGs are written.
		rolling_window: Optional integer to apply a moving average to the lines.

	Returns:
		List[Path]: Paths of the saved figures.
	"""
	trainval_df = (pd.concat([trainer.train_df, trainer.val_df])
				   .sort_values("date").reset_index(drop=True))

	targets_by_dep = defaultdict(list)
	for i, feat in enumerate(trainer.target_features):
		dep_match = next((d for d in trainer.deployment_names if feat.startswith(d + "_")), None)
		if dep_match:
			targets_by_dep[dep_match].append((i, feat))
		else:
			logger.warning("Could not match feature '%s' to any deployment name.", feat)

	saved = []
	for dep, target_items in targets_by_dep.items():
		n = len(target_items)
		fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), squeeze=False)

		for row, (idx, feat) in enumerate(target_items):
			ax = axes[row, 0]
			short = feat[len(dep) + 1:]

			raw_true = y_true[:, idx]
			raw_pred = y_pred[:, idx]

			ax.plot(trainval_df["date"].values, trainval_df[feat].values,
					color="#bbbbbb", lw=0.8, label="train+val (actual)")
			ax.axvline(dates[0], color="grey", ls=":", lw=0.8)

			if rolling_window:
				ax.plot(dates, raw_true, color="steelblue", lw=0.8, alpha=0.3)
				ax.plot(dates, raw_pred, color="darkorange", lw=0.8, ls="--", alpha=0.3)
				roll_true = pd.Series(raw_true).rolling(rolling_window, min_periods=1).mean()
				roll_pred = pd.Series(raw_pred).rolling(rolling_window, min_periods=1).mean()
				ax.plot(dates, roll_true, color="steelblue", lw=1.5,
						label=f"test actual (MA {rolling_window})")
				ax.plot(dates, roll_pred, color="darkorange", lw=1.5, ls="--",
						label=f"rollout predicted (MA {rolling_window})")
			else:
				ax.plot(dates, raw_true, color="steelblue", lw=1.2, label="test (actual)")
				ax.plot(dates, raw_pred, color="darkorange", lw=1.0, ls="--",
						label="rollout (predicted)")

			mae  = float(np.mean(np.abs(raw_pred - raw_true)))
			mse  = float(np.mean((raw_pred - raw_true) ** 2))
			rmse = float(np.sqrt(mse))

			ax.set_title(
				f"{dep} — {short}  MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}",
				fontsize=9)
			ax.legend(fontsize=7, loc="upper left", ncol=3)
			ax.tick_params(labelsize=7)
			ax.set_ylabel(short, fontsize=8)

		fig.suptitle(f"{dep} — autoregressive rollout (test split)", fontsize=11)
		fig.tight_layout()

		out_dir.mkdir(parents=True, exist_ok=True)
		path = out_dir / f"{dep}_rollout.png"
		fig.savefig(path, dpi=120, bbox_inches="tight")
		plt.close(fig)
		saved.append(path)
		logger.info("Saved rollout plot: %s", path)

	return saved

def main() -> None:
	parser = argparse.ArgumentParser(description="Train a learned-simulation transition model.")
	parser.add_argument("--model", required=True, choices=["lgbm", "lstm", "varima"],
						help="Transition model family to train.")
	parser.add_argument("--config", required=True, help="Path to the trainer YAML config.")
	parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search.")
	parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (with --tune).")
	parser.add_argument("--out", default=None, help="Artifact output directory.")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

	app = Path(args.config).stem
	out_dir = Path(args.out) if args.out else ARTIFACT_ROOT / app / args.model
	out_dir.mkdir(parents=True, exist_ok=True)

	trainer = build_trainer(args.model, args.config)

	if args.tune:
		logger.info("Tuning %s with %d Optuna trials...", args.model, args.n_trials)
		trainer.tune(args.n_trials)

	logger.info("Training %s...", args.model)
	trainer.train()

	metrics = trainer.test()

	model = trainer.to_model()
	model.metadata.update({
		"app": app,
		"tuned": args.tune,
		"best_params": trainer.best_params,
		"test_metrics": {"mae": metrics["mae"], "mse": metrics["mse"],
						 "rmse": metrics["rmse"]},
	})
	model.save(out_dir)
	logger.info("Saved %s artifact to %s", args.model, out_dir)

	logger.info("Generating rollout plots...")
	y_pred, y_true, dates = trainer.rollout()
	plot_rollout(trainer, y_pred, y_true, dates, out_dir)

if __name__ == "__main__":
	main()
