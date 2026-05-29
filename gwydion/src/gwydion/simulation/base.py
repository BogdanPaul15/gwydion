from abc import ABC, abstractmethod

from typing import Optional

import pandas as pd
import numpy as np

class SimulationStrategy(ABC):
    """Base class for all simulation strategies."""

    _METRIC_KEYS = ("cpu_usage", "mem_usage", "traffic_in", "traffic_out", "latency")

    def __init__(self, **kwargs):
        """Initializes the simulation strategy and its random number generator.

        Args:
            **kwargs: Additional keyword arguments that can be passed to configure
                specific subclasses. Pass ``seed`` to make the RNG reproducible from
                construction (otherwise the registry will call ``seed()`` shortly after).
        """
        self.rng = np.random.default_rng(kwargs.get("seed"))

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the random number generator for reproducible results.

        Args:
            seed (Optional[int]): The random seed to use.
        """
        self.rng = np.random.default_rng(seed)

    def _sample(self, df: pd.DataFrame) -> pd.Series:
        idx = self.rng.integers(0, len(df))
        return df.iloc[idx]

    @abstractmethod
    def update(self, env, action) -> None:
        """Perform one simulation step, and update env.deployment_list metrics.

        Args:
            env (BaseEnv): A BaseEnv instance.
        """

    def _write_sample_to_deployments(self, env, sample: pd.Series) -> None:
        """Write a sampled CSV row back into deployment metrics and pod counts."""
        for i, name in enumerate(env.deployments_names):
            d = env.deployment_list[i]
            d.num_previous_pods = d.num_pods
            d.num_pods = int(sample[f"{name}_num_pods"])
            d.metrics["cpu_usage"] = int(sample[f"{name}_cpu_usage"])
            d.metrics["mem_usage"] = int(sample[f"{name}_mem_usage"])
            d.metrics["traffic_in"] = int(sample[f"{name}_traffic_in"])
            d.metrics["traffic_out"] = int(sample[f"{name}_traffic_out"])
            d.metrics["latency"] = float(f"{sample[f'{name}_latency']:.3f}")

        for d in env.deployment_list:
            d.update_desired_replicas()

    def _write_metrics_from_sample(self, env, sample: pd.Series) -> None:
        """Write only the resource metrics from a dataset row, leaving pod counts untouched.

        Unlike :meth:`_write_sample_to_deployments`, ``num_pods`` and
        ``num_previous_pods`` are never modified — the agent's scaling decision
        is preserved.  Use this for every step after the initial seed.

        Args:
            env (BaseEnv): A BaseEnv instance.
            sample (pd.Series): A row from the historical dataset.
        """
        predicted = {
            f"{name}_{metric}": sample[f"{name}_{metric}"]
            for name in env.deployments_names
            for metric in self._METRIC_KEYS
        }
        self._write_metrics_to_deployments(env, predicted)

    def _write_metrics_to_deployments(self, env, predicted: dict) -> None:
        """Write predicted resource metrics into deployments, leaving pods untouched.

        Unlike :meth:`_write_sample_to_deployments`, this never changes
        ``num_pods``/``num_previous_pods`` — pod counts are owned by the agent's
        scaling action — and only updates the resource metrics. Predictions are
        clipped to non-negative values; ``desired_replicas`` is then recomputed.

        Args:
            env (BaseEnv): A BaseEnv instance.
            predicted (dict): Maps ``{deployment}_{metric}`` column names to values.
        """
        for i, name in enumerate(env.deployments_names):
            d = env.deployment_list[i]
            for metric in self._METRIC_KEYS:
                key = f"{name}_{metric}"
                if key not in predicted:
                    continue
                value = max(0.0, float(predicted[key]))
                d.metrics[metric] = round(value, 3) if metric == "latency" else int(round(value))

        for d in env.deployment_list:
            d.update_desired_replicas()
