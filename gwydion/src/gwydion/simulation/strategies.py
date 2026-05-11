import logging

from typing import List
from collections import defaultdict

import pandas as pd
import numpy as np

from scipy.spatial import KDTree

from .base import SimulationStrategy
from .registry import register

logger = logging.getLogger(__name__)

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
        self._write_sample_to_deployments(env, sample)

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

        deployment_id, _action_id = env._action_adapter.decode(action)

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

        for i in others:
            name = env.deployments_names[i]
            dep = env.deployment_list[i]
            filtered = other_matched.loc[other_matched[f"{name}_num_pods"] == dep.num_pods]
            if len(filtered) > 0:
                other_matched = filtered

        diff_action = action_dep.num_pods - action_dep.num_previous_pods

        if len(other_matched) == len(data):
            # No other deployment pod count matched — branch 1
            # try diff_action only
            with_diff = data.loc[data[f"diff-{action_name}"] == diff_action]
            sample_set = with_diff if len(with_diff) > 0 else data
        else:
            # Branch 2: other pods matched — now try diffs
            both_pods = other_matched
            with_diff_action = both_pods.loc[both_pods[f"diff-{action_name}"] == diff_action]

            # Start with the tightest constraint possible (preferring with_diff_action if valid)
            candidate = with_diff_action if len(with_diff_action) > 0 else both_pods

            # try diff_other on each other deployment
            for i in others:
                name = env.deployments_names[i]
                dep = env.deployment_list[i]
                diff_other = dep.num_pods - dep.num_previous_pods

                filtered = candidate.loc[candidate[f"diff-{name}"] == diff_other]
                if len(filtered) > 0:
                    candidate = filtered

            sample_set = candidate

        sample = self._sample(sample_set)
        self._write_sample_to_deployments(env, sample)

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
        self.col_range[self.col_range == 0] = 1.0

        normalized = (raw - self.col_min) / self.col_range
        self.tree = KDTree(normalized * self.weight_vec)

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
        return normalized * self.weight_vec

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
        self._write_sample_to_deployments(env, sample)
