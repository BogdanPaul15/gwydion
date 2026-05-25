from typing import List, Tuple

import numpy as np
import pandas as pd

from gwydion.simulation.utils import (
	add_temporal_columns, temporal_feature_names)

def delta_columns(deployment_names: List[str]) -> List[str]:
	"""Returns the ``{deployment}_delta`` column names (per-deployment pod delta)."""
	return [f"{name}_delta" for name in deployment_names]

def build_transitions(deployment_names: List[str], df: pd.DataFrame) -> pd.DataFrame:
	"""Adds per-deployment pod-delta columns describing the scaling action.

	For every deployment a ``{name}_delta`` column is added, equal to the change
	in pod count to the next row (``num_pods[t+1] - num_pods[t]``) — i.e. the
	action applied between state ``t`` and ``t + 1``.

	Args:
		deployment_names (List[str]): Ordered deployment names.
		df (pd.DataFrame): Raw observation dataframe.

	Returns:
		pd.DataFrame: Chronologically sorted dataframe with added delta columns.
	"""
	df = df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df = df.sort_values("date").reset_index(drop=True)

	for name in deployment_names:
		df[f"{name}_delta"] = df[f"{name}_num_pods"].shift(-1) - df[f"{name}_num_pods"]

	# The last row of each deployment group will have a NaN delta since there's no
	# next state to compare to.
	return df.iloc[:-1].reset_index(drop=True)

def build_tabular_dataset(df: pd.DataFrame, deployment_names: List[str],
						  state_features: List[str],
						  target_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	"""Builds a flat ``(X, y)`` dataset.

	Each row contains the current state + temporal context + action with the
	next-step targets:
	``X = [state(t), temporal(t), pod_delta(t)]``, ``y = target(t + 1)``.

	Args:
		df (pd.DataFrame): Raw observation dataframe.
		deployment_names (List[str]): Ordered deployment names.
		state_features (List[str]): Ordered state (input) column names.
		target_features (List[str]): Ordered target (output) column names.

	Returns:
		Tuple[np.ndarray, np.ndarray]: ``X`` of shape
			``(N, n_state + n_temporal + n_apps)`` and ``y`` of shape
			``(N, n_target)``.
	"""
	transitions = build_transitions(deployment_names, df)
	transitions = add_temporal_columns(transitions, target_features)

	temporal_names = temporal_feature_names(target_features)
	# We'll have NaNs in the first few rows where temporal features can't be computed due to lack
	# of history, so we need to drop those rows
	transitions = transitions.dropna(subset=temporal_names).reset_index(drop=True)

	state = transitions[state_features].to_numpy(dtype=np.float64)
	temporal = transitions[temporal_names].to_numpy(dtype=np.float64)
	deltas = transitions[delta_columns(deployment_names)].to_numpy(dtype=np.float64)
	target_next = transitions[target_features].shift(-1).to_numpy(dtype=np.float64)

	x = np.concatenate([state[:-1], temporal[:-1], deltas[:-1]], axis=1)
	y = target_next[:-1]
	return x, y

def make_windows(df: pd.DataFrame, deployment_names: List[str], state_features: List[str],
				 target_features: List[str], window: int
				 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Builds sliding state windows for LSTM.

	Args:
		df (pd.DataFrame): Raw observation dataframe.
		deployment_names (List[str]): Ordered deployment names.
		state_features (List[str]): Ordered state (input) column names.
		target_features (List[str]): Ordered target (output) column names.
		window (int): Number of timesteps per input window.

	Returns:
		Tuple[np.ndarray, np.ndarray, np.ndarray]: ``X_seq`` of shape
			``(N, window, n_state)``, ``X_act`` of shape ``(N, n_apps)`` and
			``y`` of shape ``(N, n_target)``.
	"""
	transitions = build_transitions(deployment_names, df)

	state = transitions[state_features].to_numpy(dtype=np.float64)
	deltas = transitions[delta_columns(deployment_names)].to_numpy(dtype=np.float64)
	target = transitions[target_features].to_numpy(dtype=np.float64)

	seq, act, y = [], [], []
	for t in range(window - 1, len(transitions) - 1):
		seq.append(state[t - window + 1: t + 1])
		act.append(deltas[t])
		y.append(target[t + 1])

	return np.asarray(seq), np.asarray(act), np.asarray(y)

def make_endog_exog(df: pd.DataFrame, deployment_names: List[str],
					target_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	"""Builds the endogenous/exogenous arrays for the VARIMA trainer.

	Args:
		df (pd.DataFrame): Raw observation dataframe.
		deployment_names (List[str]): Ordered deployment names.
		target_features (List[str]): Ordered target (output) column names.

	Returns:
		Tuple[np.ndarray, np.ndarray]: ``endog`` of shape ``(T, n_target)``
			(the metric series) and ``exog`` of shape ``(T, n_apps)`` (pod counts).
	"""
	ordered = df.copy()
	ordered["date"] = pd.to_datetime(ordered["date"])
	ordered = ordered.sort_values("date").reset_index(drop=True)

	endog = ordered[target_features].to_numpy(dtype=np.float64)
	exog = ordered[[f"{n}_num_pods" for n in deployment_names]].to_numpy(dtype=np.float64)
	return endog, exog
