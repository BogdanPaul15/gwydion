import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy

logger = logging.getLogger(__name__)

EPISODE_COLS = [
	"phase", "run_id", "env_rank", "episode_idx",
	"reward", "length", "wall_time",
	"avg_pods", "avg_latency", "execution_time",
	"action_stats",
]

class EpisodeMetricsWriter:
	"""Buffers per-episode metrics and flushes them to a single CSV.

	Lives in the **main** process (the SB3 callback runs there, not inside
	SubprocVecEnv workers), so concurrent appends from parallel envs are not
	an issue — every worker funnels metrics through the same writer via the
	vectorized ``infos`` list.

	Args:
		out_path: Destination CSV path. Parent directory is created if missing.
		phase: ``"train"`` or ``"test"`` - written into every row for join keys.
		run_id: Free-form identifier (e.g. ``"redis_cost_42"``) shared across
			episodes of the same run; combined with ``phase`` it uniquely tags
			a run.
		flush_every: Buffer threshold before a CSV append. Keeps writes cheap
			but means at most ``flush_every - 1`` episodes can be lost on crash.
	"""

	def __init__(self, out_path: Path, phase: str, run_id: str,
				 flush_every: int = 1):
		self.out_path = Path(out_path)
		self.out_path.parent.mkdir(parents=True, exist_ok=True)
		self.phase = phase
		self.run_id = run_id
		self.flush_every = flush_every

		self._buffer: List[dict] = []
		self._counts_per_env: Dict[int, int] = {}
		self._all_records: List[dict] = []

	def record(self, info: dict, env_rank: int) -> None:
		"""Records one completed episode from an env ``info`` dict.

		Expects ``info["episode"]`` (set by SB3's ``Monitor`` wrapper) and
		the extras the env adds at episode end (``avg_pods``, ``avg_latency``,
		``execution_time``, ``action_stats``).
		"""
		ep = info.get("episode", {})
		idx = self._counts_per_env.get(env_rank, 0)
		self._counts_per_env[env_rank] = idx + 1

		row = {
			"phase": self.phase,
			"run_id": self.run_id,
			"env_rank": env_rank,
			"episode_idx": idx,
			"reward": float(ep.get("r", 0.0)),
			"length": int(ep.get("l", 0)),
			"wall_time": float(ep.get("t", 0.0)),
			"avg_pods": float(info.get("avg_pods", 0.0)),
			"avg_latency": float(info.get("avg_latency", 0.0)),
			"execution_time": float(info.get("execution_time", 0.0)),
			"action_stats": json.dumps(list(info.get("action_stats", []))),
		}
		self._buffer.append(row)
		self._all_records.append(row)

		if len(self._buffer) >= self.flush_every:
			self.flush()

	def flush(self) -> None:
		"""Appends any buffered rows to the CSV. Idempotent on empty buffer."""
		if not self._buffer:
			return
		df = pd.DataFrame(self._buffer, columns=EPISODE_COLS)
		header = not self.out_path.exists()
		df.to_csv(self.out_path, mode="a", index=False, header=header)
		self._buffer.clear()

	def write_summary(self, summary_path: Optional[Path] = None) -> dict:
		"""Flushes the buffer and writes a ``summary.json``.

		The summary aggregates rewards, episode lengths, avg_pods, avg_latency
		and execution_time as ``{mean, std, min, max}`` plus a total episode
		count.
		"""
		self.flush()
		summary_path = Path(summary_path or self.out_path.parent / "summary.json")

		if not self._all_records:
			summary = {"phase": self.phase, "run_id": self.run_id, "n_episodes": 0}
		else:
			df = pd.DataFrame(self._all_records)
			summary = {
				"phase": self.phase,
				"run_id": self.run_id,
				"n_episodes": int(len(df)),
				"metrics": {
					col: {
						"mean": float(df[col].mean()),
						"std":  float(df[col].std(ddof=0)),
						"min":  float(df[col].min()),
						"max":  float(df[col].max()),
					}
					for col in ["reward", "length", "wall_time",
								"avg_pods", "avg_latency", "execution_time"]
				},
			}

		summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
		return summary

class EpisodeCallback(BaseCallback):
	"""SB3 callback that forwards completed episodes to an :class:`EpisodeMetricsWriter`.

	Each step inspects the vectorized ``infos`` list and records any entry
	where ``"episode"`` is present (added by ``Monitor`` on the auto-reset
	step). Safe under ``n_envs > 1`` since the writer lives in the main
	process.
	"""

	def __init__(self, writer: EpisodeMetricsWriter):
		super().__init__(verbose=0)
		self.writer = writer

	def _on_step(self) -> bool:
		for env_rank, info in enumerate(self.locals.get("infos", [])):
			if "episode" in info:
				self.writer.record(info, env_rank=env_rank)
		return True

	def _on_training_end(self) -> None:
		self.writer.write_summary()

class StepObsWriter:
	"""Buffers per-step observations from all envs and flushes them to a CSV.

	Records the raw observation vector alongside the step number, env rank,
	and per-step latency from the ``info`` dict. Useful for diagnosing what
	the agent observed across an episode.

	Args:
		out_path: Destination CSV path. Parent is created if missing.
		feature_names: Column names for the observation vector. Defaults to
			``obs_0, obs_1, ...`` when omitted.
		flush_every: Row buffer size before a CSV append.
	"""

	def __init__(self, out_path: Path,
				 feature_names: Optional[List[str]] = None,
				 flush_every: int = 25):
		self.out_path = Path(out_path)
		self.out_path.parent.mkdir(parents=True, exist_ok=True)
		self._feature_names = feature_names
		self.flush_every = flush_every
		self._buffer: List[dict] = []
		self._columns: Optional[List[str]] = None

	def _ensure_columns(self, n_features: int,
						 extra: Optional[Dict[str, Any]] = None) -> None:
		if self._columns is not None:
			return
		feat = self._feature_names or [f"obs_{i}" for i in range(n_features)]
		self._columns = ["step", "env_rank", "latency"] + list(feat)
		if extra:
			self._columns += list(extra.keys())

	def record(self, step: int, env_rank: int,
			   obs: Any, latency: float,
			   extra: Optional[Dict[str, Any]] = None) -> None:
		"""Appends one observation row to the buffer."""
		self._ensure_columns(len(obs), extra)
		row: Dict[str, Any] = {
			"step": step, "env_rank": env_rank, "latency": latency,
		}
		obs_cols = self._columns[3: 3 + len(obs)]
		for name, val in zip(obs_cols, obs):
			row[name] = float(val)
		if extra:
			row.update(extra)
		self._buffer.append(row)
		if len(self._buffer) >= self.flush_every:
			self.flush()

	def flush(self) -> None:
		"""Appends any buffered rows to the CSV. Idempotent on empty buffer."""
		if not self._buffer:
			return
		df = pd.DataFrame(self._buffer, columns=self._columns)
		header = not self.out_path.exists()
		df.to_csv(self.out_path, mode="a", index=False, header=header)
		self._buffer.clear()

class StepObsCallback(BaseCallback):
	"""SB3 callback that reads raw observations from the env and forwards
	them to a :class:`StepObsWriter` after each step."""

	def __init__(self, writer: StepObsWriter):
		super().__init__(verbose=0)
		self.writer = writer

	def _on_step(self) -> bool:
		infos = self.locals.get("infos", [])
		raw_obs_list = self.training_env.get_attr("last_obs")
		for rank, (obs_row, info) in enumerate(zip(raw_obs_list, infos)):
			if obs_row is None:
				continue
			EXTRA = ("_desired_replicas", "_traffic_in", "_traffic_out")
			extra = {k: float(v) for k, v in info.items()
					 if any(k.endswith(s) for s in EXTRA)}
			self.writer.record(
				step=self.num_timesteps,
				env_rank=rank,
				obs=obs_row,
				latency=float(info.get("latency", 0.0)),
				extra=extra or None,
			)
		return True

	def _on_training_end(self) -> None:
		self.writer.flush()

class OptunaPruningCallback(BaseCallback):
	"""Periodically evaluates the model during an Optuna trial and prunes
	underperforming trials early based on intermediate results.

	Args:
		trial: The active Optuna trial being evaluated.
		eval_env: A VecNormalize-wrapped env used for evaluation.
			Must have ``training=False`` so statistics are not updated during eval.
		n_eval_episodes: Number of episodes to run per evaluation.
		eval_freq: Evaluate every ``eval_freq`` training steps.
		maskable: If True, uses MaskablePPO evaluation.
	"""

	def __init__(self, trial: optuna.Trial, eval_env: VecNormalize,
				 n_eval_episodes: int = 5, eval_freq: int = 10_000,
				 maskable: bool = False):
		super().__init__(verbose=0)
		self.trial = trial
		self.eval_env = eval_env
		self.n_eval_episodes = n_eval_episodes
		self.eval_freq = eval_freq
		self.maskable = maskable

	def _on_step(self) -> bool:
		if self.n_calls % self.eval_freq != 0:
			return True

		train_env = self.model.get_env()
		self.eval_env.obs_rms = train_env.obs_rms
		self.eval_env.ret_rms = train_env.ret_rms

		evaluator = maskable_evaluate_policy if self.maskable else evaluate_policy
		mean_reward, _ = evaluator(
			self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)

		self.trial.report(mean_reward, step=self.n_calls)
		if self.trial.should_prune():
			raise optuna.TrialPruned()
		return True
