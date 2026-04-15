from abc import ABC, abstractmethod

import pandas as pd

class SimulationStrategy(ABC):
    """Base class for all simulation strategies."""

    @abstractmethod
    def update(self, env) -> None:
        """Perform one simulation step, and update env.deployment_list metrics.

        Args:
            env (BaseEnv): A BaseEnv instance.
        """

    @staticmethod
    def _write_sample_to_deployments(env, sample: pd.Series) -> None:
        """Write a sampled CSV row back into deployment metrics."""
        for i, name in enumerate(env.deployments_names):
            d = env.deployment_list[i]
            d.metrics["cpu_usage"] = int(sample[f"{name}_cpu_usage"])
            d.metrics["mem_usage"] = int(sample[f"{name}_mem_usage"])
            d.metrics["received_traffic"] = int(sample[f"{name}_traffic_in"])
            d.metrics["transmit_traffic"] = int(sample[f"{name}_traffic_out"])
            d.metrics["latency"] = float(f"{sample[f'{name}_latency']:.3f}")

        for d in env.deployment_list:
            d.update_desired_replicas()

    @staticmethod
    def _sample_initial_step(env) -> pd.Series:
        """Random sample to initialize deployment state."""
        sample = env.df.sample(n=1).iloc[0]

        for i, name in enumerate(env.deployments_names):
            env.deployment_list[i].num_pods = int(sample[f"{name}_num_pods"])
            env.deployment_list[i].num_previous_pods = int(sample[f"{name}_num_pods"])

        return sample
