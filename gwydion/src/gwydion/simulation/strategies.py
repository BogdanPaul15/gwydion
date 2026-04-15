import logging

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

    def update(self, env) -> None:
        if env.current_step == 1:
            sample = self._sample_initial_step(env)
            self._write_sample_to_deployments(env, sample)
            return

        pods = [d.num_pods for d in env.deployment_list]
        diff = [d.num_pods - d.num_previous_pods for d in env.deployment_list]
        data = env.df

        for i in range(env.num_apps):
            name = env.deployments_names[i]
            snapshot = data

            # Exact pod count and diff count match
            exact = data.loc[(data[f"{name}_num_pods"] == pods[i]) &
                             (data[f"diff-{name}"] == diff[i])]

            if len(exact) > 0:
                data = exact
                continue

            # Pod count only
            pods_only = data.loc[data[f"{name}_num_pods"] == pods[i]]

            if len(pods_only) > 0:
                logger.debug("[Step %d] | No exact match for %s (pods=%d, diff=%d), relaxing diff filter",
                                env.current_step, name, pods[i], diff[i])
                data = pods_only
                continue

            logger.warning("[Step %d] | No pod match for %s (pods=%d) within current candidates. "
                    "Keeping previous filter state (%d rows).",
                    env.current_step, name, pods[i], len(snapshot))
            data = snapshot

        if len(data) == 0:
            logger.warning("[Step %d] All filters exhausted. Sampling full dataset.",
                           env.current_step)
            data = env.df

        sample = data.sample(n=1).iloc[0]
        self._write_sample_to_deployments(env, sample)
