import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class LiveMetricsExporter:
    """Publishes the agent's per-step state as Prometheus gauges.

    Attributes:
        port: TCP port the metrics HTTP server listens on (default ``8000``).
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self._prev_pods: Dict[str, int] = {}
        self._episode_start: float = time.monotonic()
        self._last_episode: int = 0
        self._started = False
        self._build_metrics()

    def _build_metrics(self) -> None:
        try:
            from prometheus_client import Counter, Gauge
        except ImportError as e:
            raise ImportError(
                "Live metrics require the 'prometheus-client' package. "
                "Install it with: pip install prometheus-client"
            ) from e

        dep = ["deployment"]
        self.g_pods = Gauge("gwydion_pods", "Current pod replicas per deployment", dep)
        self.g_desired = Gauge("gwydion_desired_replicas",
                               "Heuristic desired replicas per deployment", dep)
        self.g_cpu = Gauge("gwydion_cpu_usage_millicores",
                           "Observed aggregated CPU usage per deployment (millicores)", dep)
        self.g_mem = Gauge("gwydion_mem_usage_mb",
                           "Observed aggregated memory usage per deployment (MB)", dep)
        self.c_scale_events = Counter("gwydion_scale_events",
                                      "Total scaling actions applied per deployment", dep)

        self.g_total_pods = Gauge("gwydion_total_pods", "Total pods across all deployments")
        self.g_latency = Gauge("gwydion_latency_ms", "Target deployment latency (ms)")
        self.g_step_reward = Gauge("gwydion_step_reward", "Reward for the most recent step")
        self.g_episode_reward = Gauge("gwydion_episode_reward",
                                      "Cumulative reward in the current episode")
        self.g_step = Gauge("gwydion_step", "Current step within the episode")
        self.g_episode = Gauge("gwydion_episode", "Completed episode count")
        self.g_none_counter = Gauge("gwydion_none_counter",
                                    "Consecutive DoNothing actions")
        self.g_episode_duration = Gauge("gwydion_episode_duration_seconds",
                                        "Wall-clock duration of each completed episode",
                                        ["episode"])

    def start(self) -> None:
        """Starts the metrics HTTP server (idempotent)."""
        if self._started:
            return
        from prometheus_client import start_http_server
        start_http_server(self.port)
        self._started = True
        logger.info("Live metrics exporter listening on :%d/metrics", self.port)

    @staticmethod
    def _attr(env, name):
        """Reads an attribute off the (possibly vectorised) env's first worker."""
        if hasattr(env, "get_attr"):
            return env.get_attr(name)[0]
        return getattr(env, name)

    def update(self, env, step_reward: Optional[float] = None) -> None:
        """Refreshes all gauges from the environment's current state.

        Args:
            env: The (vectorised) test environment wrapping a single ``BaseEnv``.
            step_reward: The reward returned by the most recent ``step`` call.
        """
        deployments = self._attr(env, "deployment_list")

        total_pods = 0
        for d in deployments:
            pods = int(d.num_pods)
            total_pods += pods
            self.g_pods.labels(d.name).set(pods)
            self.g_desired.labels(d.name).set(float(d.desired_replicas))
            self.g_cpu.labels(d.name).set(float(d.metrics.get("cpu_usage", 0)))
            self.g_mem.labels(d.name).set(float(d.metrics.get("mem_usage", 0)))

            prev = self._prev_pods.get(d.name, pods)
            direction = (pods > prev) - (pods < prev)
            if direction != 0:
                self.c_scale_events.labels(d.name).inc()
            self._prev_pods[d.name] = pods

        self.g_total_pods.set(total_pods)

        target_id = self._attr(env, "_cfg")["env"]["target_id"]
        self.g_latency.set(float(deployments[target_id].metrics.get("latency", 0.0)))

        if step_reward is not None:
            self.g_step_reward.set(float(step_reward))
        self.g_episode_reward.set(float(self._attr(env, "total_reward")))
        self.g_step.set(int(self._attr(env, "current_step")))

        episode = int(self._attr(env, "episode_count"))
        self.g_episode.set(episode)
        self.g_none_counter.set(int(self._attr(env, "none_counter")))

        if episode > self._last_episode:
            duration = time.monotonic() - self._episode_start
            self.g_episode_duration.labels(episode=str(episode)).set(duration)
            self._episode_start = time.monotonic()
            self._last_episode = episode
