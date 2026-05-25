from typing import List, Tuple

import numpy as np
import pandas as pd

TEMPORAL_LAGS: Tuple[int, ...] = (1, 2, 3)
TEMPORAL_ROLLING_WINDOWS: Tuple[int, ...] = (5, 15)
TEMPORAL_DEVIATION_WINDOW: int = 15

def temporal_max_lookback() -> int:
	"""Returns the largest past-step index required by any temporal feature.

	A rolling/lag of size ``window`` needs ``window`` values strictly before the current
	step, so the model needs a history buffer of length
	``temporal_max_lookback() + 1`` (the +1 is the current state).
	"""
	return max(*TEMPORAL_LAGS, *TEMPORAL_ROLLING_WINDOWS, TEMPORAL_DEVIATION_WINDOW)

def temporal_feature_names(target_features: List[str]) -> List[str]:
	"""Returns the temporal feature column names in order.

	The order returned here must match the order produced by both
	:func:`add_temporal_columns` (training) and
	:func:`compute_temporal_features` (inference).
	"""
	names: List[str] = []
	for feature in target_features:
		for lag in TEMPORAL_LAGS:
			names.append(f"{feature}_lag{lag}")
		for window in TEMPORAL_ROLLING_WINDOWS:
			names.append(f"{feature}_rmean{window}")
			names.append(f"{feature}_rstd{window}")
		names.append(f"{feature}_rdev{TEMPORAL_DEVIATION_WINDOW}")
	return names

def add_temporal_columns(df: pd.DataFrame, target_features: List[str]) -> pd.DataFrame:
	"""Adds lag/rolling columns for each target feature (training).

	Rolling statistics are computed over the **past** ``window`` values only
	(``.shift(1).rolling(window=window)``) — the current step is never part of its own
	rolling stat, so the model can't trivially read the label from a feature.
	The first ``temporal_max_lookback()`` rows will contain NaNs; the caller
	is expected to drop them.

	Args:
		df (pd.DataFrame): Chronologically sorted dataframe.
		target_features (List[str]): Columns to compute temporal features over.

	Returns:
		pd.DataFrame: Copy of ``df`` with the additional temporal columns.
	"""
	df = df.copy()
	for feature in target_features:
		for lag in TEMPORAL_LAGS:
			df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
		for window in TEMPORAL_ROLLING_WINDOWS:
			past = df[feature].shift(1).rolling(window=window, min_periods=window)
			df[f"{feature}_rmean{window}"] = past.mean()
			df[f"{feature}_rstd{window}"] = past.std()
		deviation_window = TEMPORAL_DEVIATION_WINDOW
		rmean_dev = df[feature].shift(1).rolling(window=deviation_window, min_periods=deviation_window).mean()
		df[f"{feature}_rdev{deviation_window}"] = df[feature] - rmean_dev
	return df

def compute_temporal_features(history: np.ndarray, target_indices: List[int]) -> np.ndarray:
	"""Computes temporal features from a live state-history buffer (inference).

	``history[-1]`` is treated as the current state. Rolling/lag values look at
	``history[-window-1:-1]`` (the ``window`` states strictly before the current one),
	which exactly mirrors the ``.shift(1).rolling(window=window)`` semantics used in
	:func:`add_temporal_columns`.

	Args:
		history (np.ndarray): State history of shape ``(h, n_state)`` with
			``h >= temporal_max_lookback() + 1``.
		target_indices (List[int]): Indices of target columns inside
			``state_features`` — i.e. which columns of ``history`` are the
			metrics being predicted.

	Returns:
		np.ndarray: Flat feature vector in the same order as
				:func:`temporal_feature_names`.
	"""
	target_history = history[:, target_indices]
	features: List[float] = []
	deviation_window = TEMPORAL_DEVIATION_WINDOW

	for i in range(target_history.shape[1]):
		col = target_history[:, i]
		current = col[-1]
		for lag in TEMPORAL_LAGS:
			features.append(float(col[-1-lag]))
		for window in TEMPORAL_ROLLING_WINDOWS:
			past_window = col[-window-1:-1]
			features.append(float(past_window.mean()))
			features.append(float(past_window.std(ddof=1)))
		past_deviation = col[-deviation_window-1:-1]
		features.append(float(current - past_deviation.mean()))

	return np.asarray(features, dtype=np.float64)
