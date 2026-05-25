import logging

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from scipy.spatial import KDTree

from .base import SimulationStrategy
from .registry import register
from .models import load_simulator_model

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

@register("default")
class DefaultSimulationStrategy(SimulationStrategy):
    """Hierarchical filtering strategy.

    For each deployment in order, filters the dataset on:
    - exact pod count and diff count *if rows exist, keep filtering 
    - pod count only (relax diff) *if rows exist, keep filtering
    - neither matches *keep previous filter state

    If all filters produce an empty set, falls back to the full dataset.

    It treats all deployments equally, iterating through them in 
    index order.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def update(self, env, _action) -> None:
        if env.current_step == 1:
            sample = self._sample(env.df)
            self._write_sample_to_deployments(env, sample)
            return

        pods = [d.num_pods for d in env.deployment_list]
        diff = [d.num_pods - d.num_previous_pods for d in env.deployment_list]
        data = env.df

        for idx, name in enumerate(env.deployments_names):
            snapshot = data

            # Exact pod count and diff count match
            exact = data.loc[(data[f"{name}_num_pods"] == pods[idx]) &
                             (data[f"diff-{name}"] == diff[idx])]

            if len(exact) > 0:
                data = exact
                continue

            # Pod count only
            pods_only = data.loc[data[f"{name}_num_pods"] == pods[idx]]

            if len(pods_only) > 0:
                logger.debug("[Step %d] | No exact match for %s (pods=%d, diff=%d), relaxing diff filter",
                                env.current_step, name, pods[idx], diff[idx])
                data = pods_only
                continue

            # No match within current candidates (keep previous filter state)
            logger.warning("[Step %d] | No pod match for %s (pods=%d) within current candidates. "
                    "Keeping previous filter state (%d rows).",
                    env.current_step, name, pods[idx], len(snapshot))
            data = snapshot

        if len(data) == 0:
            logger.warning("[Step %d] All filters exhausted. Sampling full dataset.",
                           env.current_step)
            data = env.df

        sample = self._sample(data)
        self._write_metrics_from_sample(env, sample)

@register("action_aware")
class ActionAwareSimulationStrategy(SimulationStrategy):
    """TODO"""
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, env, action) -> None:
        if env.current_step == 1:
            sample = self._sample(env.df)
            self._write_sample_to_deployments(env, sample)
            return

        # Primary deployment: first one with a non-noop action this step.
        # Falls back to deployment 0 when every action is a no-op.
        action_pairs = env._action_adapter.decode(action)
        primary = next(
            ((dep_id, act_id) for dep_id, act_id in action_pairs
             if not env._actions[act_id].is_noop),
            action_pairs[0],
        )
        deployment_id, _ = primary

        action_name = env.deployments_names[deployment_id]
        action_dep = env.deployment_list[deployment_id]

        # Acted deployment pod count only
        data = env.df.loc[env.df[f"{action_name}_num_pods"] == action_dep.num_pods]

        if len(data) == 0:
            sample = self._sample(env.df)
            self._write_sample_to_deployments(env, sample)
            return

        # Try matching all other deployments on pod count
        others = [i for i in range(env.num_apps) if i != deployment_id]
        other_matched = data
        narrowed = False

        for i in others:
            name = env.deployments_names[i]
            dep = env.deployment_list[i]
            filtered = other_matched.loc[other_matched[f"{name}_num_pods"] == dep.num_pods]
            if len(filtered) > 0:
                other_matched = filtered
                narrowed = True

        diff_action = action_dep.num_pods - action_dep.num_previous_pods

        if not narrowed:
            # No other deployment pod count matched — try diff on primary only
            with_diff = data.loc[data[f"diff-{action_name}"] == diff_action]
            sample_set = with_diff if len(with_diff) > 0 else data
        else:
            # Other pods matched — tighten with diffs
            with_diff_action = other_matched.loc[
                other_matched[f"diff-{action_name}"] == diff_action
            ]
            candidate = with_diff_action if len(with_diff_action) > 0 else other_matched

            for i in others:
                name = env.deployments_names[i]
                dep = env.deployment_list[i]
                diff_other = dep.num_pods - dep.num_previous_pods
                filtered = candidate.loc[candidate[f"diff-{name}"] == diff_other]
                if len(filtered) > 0:
                    candidate = filtered

            sample_set = candidate

        sample = self._sample(sample_set)
        self._write_metrics_from_sample(env, sample)

@register("knn")
class KNNSimulationStrategy(SimulationStrategy):
    """Weighted K-Nearest-Neighbor simulation.
    
    Encodes current deployment state as a vector, queries a KD-tree
    built from the historical dataset, and samples from the k nearest
    neighbors weighted by inverse distance. 
    """

    def __init__(self, **kwargs):
        """TODO"""
        super().__init__(**kwargs)
        df = kwargs.get("df")
        if df is None:
            raise ValueError("KNNSimulationStrategy requires a 'df' (pandas DataFrame) in kwargs.")
        self.deployments = kwargs.get("deployment_names", [])
        config = kwargs.get("config", {})
        self.k = config.get("k", 10)
        self.distance_weight_power = config.get("distance_weight_power", 2.0)

        self.weights = config.get("resources", {
            "num_pods": 5.0,
            "diff": 5.0,
            "cpu_usage": 1.5,
            "mem_usage": 1.0
        })

        self.state_columns = []
        feature_weights = []
        for deployment in self.deployments:
            for resource, weight in self.weights.items():
                patterns = [f"{deployment}_{resource}", f"{resource}-{deployment}"]

                for p in patterns:
                    if p in df.columns:
                        self.state_columns.append(p)
                        feature_weights.append(float(weight))
                        break

        if not self.state_columns:
            raise ValueError("KNNSimulationStrategy: no matching columns found in dataset for given weights/deployments.")

        self.weight_vec = np.array(feature_weights)

        raw = df[self.state_columns].values.astype(np.float64)
        self.col_min = raw.min(axis=0)
        self.col_max = raw.max(axis=0)
        self.col_range = self.col_max - self.col_min

        # Constant columns carry no information
        self.active_mask = self.col_range > 0
        self.col_range = np.where(self.active_mask, self.col_range, 1.0)
        effective_weights = np.where(self.active_mask, self.weight_vec, 0.0)

        normalized = (raw - self.col_min) / self.col_range
        self.tree = KDTree(normalized * effective_weights)
        self._effective_weights = effective_weights

    def _encode_state(self, env) -> np.ndarray:
        """TODO"""
        values = []

        for i, dep in enumerate(self.deployments):
            deployment = env.deployment_list[i]

            for resource in self.weights.keys():
                patterns = [f"{dep}_{resource}", f"{resource}-{dep}"]
                active_col = next((p for p in patterns if p in self.state_columns), None)

                if active_col is None:
                    continue

                if resource == "num_pods":
                    val = float(deployment.num_pods)
                elif resource == "diff":
                    val = float(deployment.num_pods - deployment.num_previous_pods)
                else:
                    val = float(deployment.metrics.get(resource, 0))

                values.append(val)

        raw = np.array(values, dtype=np.float64)
        if len(raw) != len(self.weight_vec):
            raise ValueError(
                f"_encode_state produced vector of length {len(raw)} "
                f"but tree expects {len(self.weight_vec)}. "
                f"State columns: {self.state_columns}"
            )

        normalized = (raw - self.col_min) / self.col_range
        return normalized * self._effective_weights

    def _query(self, env) -> pd.Series:
        query_vec = self._encode_state(env)

        distances, indices = self.tree.query(query_vec, k=self.k)

        distances = np.atleast_1d(distances).astype(np.float64)
        indices = np.atleast_1d(indices)

        epsilon = 1e-8
        weights = 1.0 / (distances + epsilon) ** self.distance_weight_power
        probabilities = weights / weights.sum()

        chosen_idx = self.rng.choice(indices, p=probabilities)
        return env.df.iloc[chosen_idx]

    def update(self, env, action) -> None:
        if env.current_step == 1:
            sample = self._sample(env.df)
            self._write_sample_to_deployments(env, sample)
            return

        sample = self._query(env)
        self._write_metrics_from_sample(env, sample)

@register("learned")
class LearnedSimulationStrategy(SimulationStrategy):
    """Model-driven simulation strategy.

    Instead of sampling the historical dataset, this strategy drives the
    environment with an offline-trained transition model (a
    :class:`~gwydion.simulation.models.SimulatorModel`). The first step seeds a
    random real state from the dataset; every subsequent step predicts the next
    resource metrics from the recent state history and the agent's scaling
    action, then feeds the prediction back in — an autoregressive rollout.

    Pod counts stay owned by the agent's action: only resource metrics are
    predicted and written, after which ``desired_replicas`` is recomputed.
    """

    def __init__(self, **kwargs):
        """Loads the pretrained model and prepares state-vector bookkeeping.

        Args:
            **kwargs: Must contain ``df`` (historical dataset, for seeding),
                ``deployment_names`` and ``config`` (with a ``model_path`` key).
        """
        super().__init__(**kwargs)

        config = kwargs.get("config", {})
        self.df = kwargs.get("df")
        self.deployments = kwargs.get("deployment_names", [])

        if self.df is None:
            raise ValueError("LearnedSimulationStrategy requires a 'df' in kwargs.")

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError(
                "LearnedSimulationStrategy requires 'model_path' in the "
                "simulation_strategy config block."
            )
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        self.model = load_simulator_model(resolved)
        logger.info("Loaded '%s' simulator model from %s", self.model.model_type, resolved)

        # Map each model state feature to a (deployment index, feature name) pair.
        self._state_spec = []
        for col in self.model.state_features:
            owner = max((n for n in self.deployments if col.startswith(f"{n}_")),
                        key=len, default=None)
            if owner is None:
                raise ValueError(f"State feature '{col}' matches no known deployment.")
            self._state_spec.append((self.deployments.index(owner), col[len(owner) + 1:]))

        self._history: List[np.ndarray] = []
        self._prev_pods: List[int] = []

    def _state_vec_from_row(self, row: pd.Series) -> np.ndarray:
        """Builds a model state vector from a dataset row."""
        return np.array([float(row[col]) for col in self.model.state_features],
                        dtype=np.float64)

    def _state_vec_from_env(self, env) -> np.ndarray:
        """Builds a model state vector from the live deployment state."""
        values = []
        for dep_idx, feature in self._state_spec:
            d = env.deployment_list[dep_idx]
            if feature == "num_pods":
                values.append(float(d.num_pods))
            elif feature == "desired_replicas":
                values.append(float(d.desired_replicas))
            else:
                values.append(float(d.metrics.get(feature, 0.0)))
        return np.array(values, dtype=np.float64)

    def _seed_initial_state(self, env) -> None:
        """Seeds the rollout from a random contiguous slice of real observations."""
        history_len = self.model.required_history
        start = int(self.rng.integers(0, len(self.df) - history_len + 1))
        window = self.df.iloc[start:start + history_len]

        self._write_sample_to_deployments(env, window.iloc[-1])
        self._history = [self._state_vec_from_row(window.iloc[k])
                         for k in range(history_len)]
        self._prev_pods = [d.num_pods for d in env.deployment_list]

    def update(self, env, action) -> None:
        if env.current_step == 1:
            self._seed_initial_state(env)
            return

        # Actual per-deployment pod delta applied this step (handles constraints).
        action_delta = np.array(
            [env.deployment_list[i].num_pods - self._prev_pods[i]
             for i in range(env.num_apps)], dtype=np.float64)

        history_len = self.model.required_history
        history = np.array(self._history[-history_len:], dtype=np.float64)

        prediction = self.model.predict_next(history, action_delta)
        predicted = dict(zip(self.model.target_features, prediction))
        self._write_metrics_to_deployments(env, predicted)

        self._history.append(self._state_vec_from_env(env))
        self._prev_pods = [d.num_pods for d in env.deployment_list]
