from abc import ABC, abstractmethod

from typing import Optional

import pandas as pd
import numpy as np

class SimulationStrategy(ABC):
    """Base class for all simulation strategies."""

    def __init__(self, **kwargs):
        """TODO"""
        self.rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:
        """TODO"""
        self.rng = np.random.default_rng(seed)

    def _sample(self, df: pd.DataFrame) -> pd.Series:
        idx = self.rng.integers(0, len(df))
        return df.iloc[idx]

    @abstractmethod
    def update(self, env) -> None:
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
