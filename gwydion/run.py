import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from gwydion.arena import Arena
from gwydion.envs import OnlineBoutique, Redis
from gwydion.rewards import CostStrategy, SmoothCostStrategy, LatencyStrategy, MultiObjectiveStrategy

logger = logging.getLogger(__name__)

USE_CASES = {
	"redis":          (Redis,          "configs/redis.yaml",          0),
	"onlineboutique": (OnlineBoutique, "configs/online_boutique.yaml", 9),
}

LATENCY_THRESHOLDS = {
	"redis":          250.0,
	"onlineboutique": 3000.0,
}

def build_reward_strategy(goal: str, use_case: str,
						   cost_weight: float = 1.0, latency_weight: float = 1.0):
	target_id = USE_CASES[use_case][2]
	threshold = LATENCY_THRESHOLDS[use_case]

	if goal == "cost":
		return CostStrategy()
	if goal == "smooth_cost":
		return SmoothCostStrategy()
	if goal == "latency":
		return LatencyStrategy(target_id=target_id, threshold=threshold)
	if goal == "multi":
		return MultiObjectiveStrategy(objectives=[
			(CostStrategy(), cost_weight),
			(LatencyStrategy(target_id=target_id, threshold=threshold), latency_weight),
		])
	raise ValueError(f"Unknown --goal '{goal}'.")

def load_hyperparams(path: str, expected_strategy: str | None = None) -> dict:
	"""Loads a best_params.json (output of ``arena.tune()``) and returns the
	raw sampled params."""
	data = json.loads(Path(path).read_text(encoding="utf-8"))
	saved = data.get("reward_strategy")
	if saved and expected_strategy and saved != expected_strategy:
		print(
			f"WARNING: hyperparams were tuned with '{saved}' but you are "
			f"training with '{expected_strategy}'. Pass --goal to match.",
			file=sys.stderr,
		)
	return data.get("best_params", data)

def parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Arena entrypoint.")
	p.add_argument("--phase", required=True, choices=["tune", "train", "test"])
	p.add_argument("--alg", required=True,
				   choices=["ppo", "recurrent_ppo", "maskable_ppo", "trpo", "a2c"])
	p.add_argument("--use-case", required=True, choices=list(USE_CASES))
	p.add_argument("--goal", default="cost",
				   choices=["cost", "smooth_cost", "latency", "multi"])
	p.add_argument("--cost-weight", type=float, default=1.0,
				   help="Cost objective weight (--goal multi only).")
	p.add_argument("--latency-weight", type=float, default=1.0,
				   help="Latency objective weight (--goal multi only).")
	p.add_argument("--config",
				   help="Override the default config for this use case "
						"(default: configs/{use-case}.yaml).")
	p.add_argument("--n-envs", type=int, default=1,
				   help="Parallel envs for tune/train.")
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--output-dir", default="arena_results")

	# tune
	p.add_argument("--n-trials", type=int, default=20)
	p.add_argument("--tune-steps", type=int, default=200_000)
	p.add_argument("--n-eval-episodes", type=int, default=10)
	p.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna trials.")
	p.add_argument("--resume-tune",
				   help="Path to an existing tune_* directory to resume from. "
						"Skips timestamp dir creation; reuses optuna_study.db.")

	# train
	p.add_argument("--total-steps", type=int, default=500_000)
	p.add_argument("--save-freq", type=int, default=50_000)
	p.add_argument("--run-label", default="default")
	p.add_argument("--hyperparams",
				   help="Path to a best_params.json from a tune run.")
	p.add_argument("--resume-model",
				   help="Path to a .zip model checkpoint to resume training from.")
	p.add_argument("--resume-stats",
				   help="Path to the matching vecnormalize.pkl when resuming.")

	# test
	p.add_argument("--model", help="Path to the trained model .zip (test phase).")
	p.add_argument("--stats", help="Path to the vecnormalize.pkl (test phase).")
	p.add_argument("--n-episodes", type=int, default=100)

	p.add_argument("--record-step-obs", action="store_true",
				   help="Save raw per-step observations to step_obs.csv.")
	p.add_argument("--stochastic", action="store_true",
				   help="Sample actions from the policy instead of taking the "
						"deterministic argmax (test phase). Useful for probing "
						"less-collapsed earlier checkpoints.")
	return p

def main() -> None:
	args = parser().parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	log_path = Path(args.output_dir) / "gwydion.log"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	file_handler = logging.FileHandler(log_path)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

	root = logging.getLogger()
	root.setLevel(logging.DEBUG)
	root.handlers.clear()
	root.addHandler(file_handler)

	env_cls, default_cfg, _ = USE_CASES[args.use_case]
	config_path = args.config or default_cfg
	reward = build_reward_strategy(args.goal, args.use_case,
									cost_weight=args.cost_weight,
									latency_weight=args.latency_weight)

	arena = Arena(
		env_cls=env_cls,
		alg=args.alg,
		n_envs=args.n_envs,
		seed=args.seed,
		output_dir=args.output_dir,
	)

	if args.phase == "tune":
		arena.tune(
			config_path=config_path,
			reward_strategy=reward,
			n_trials=args.n_trials,
			tune_steps=args.tune_steps,
			n_eval_episodes=args.n_eval_episodes,
			n_jobs=args.n_jobs,
			resume_from_dir=args.resume_tune,
		)
	elif args.phase == "train":
		hp = load_hyperparams(args.hyperparams, reward.__class__.__name__) if args.hyperparams else None
		resume = ((args.resume_model, args.resume_stats)
				  if args.resume_model and args.resume_stats else None)
		arena.train(
			config_path=config_path,
			reward_strategy=reward,
			hyperparams=hp,
			total_steps=args.total_steps,
			save_freq=args.save_freq,
			run_label=args.run_label,
			resume_from=resume,
			record_step_obs=args.record_step_obs,
		)
	elif args.phase == "test":
		if not args.model:
			print("--model is required for --phase test", file=sys.stderr)
			sys.exit(2)
		arena.test(
			config_path=config_path,
			reward_strategy=reward,
			model_path=args.model,
			stats_path=args.stats,
			n_episodes=args.n_episodes,
			run_label=args.run_label,
			record_step_obs=args.record_step_obs,
			deterministic=not args.stochastic,
		)

if __name__ == "__main__":
	main()
