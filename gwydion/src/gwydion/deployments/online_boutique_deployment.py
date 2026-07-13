import math
import logging

from .deployment import Deployment
from .deployment_registry import register

logger = logging.getLogger(__name__)

@register("online_boutique")
class OnlineBoutiqueDeployment(Deployment):
    """Concrete deployment implementation for Online Boutique gym environment.

    Scales based on a weighted CPU and MEM usage, Network I/O,
    and by tracking cart specific latency.
    """

    def __init__(self, k8s, name, namespace, min_pods, max_pods,
                 cpu_request, cpu_limit, mem_request, mem_limit,
                 host=None, token=None, prometheus_url=None,
                 cpu_weight=0.7, mem_weight=0.3, threshold=0.75):
        super().__init__(k8s, name, namespace, min_pods, max_pods,
                         host=host, token=token, prometheus_url=prometheus_url)

        self.cpu_request = cpu_request
        self.mem_request = mem_request
        self.threshold = threshold

        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit

        self.cpu_target = int(self.threshold * self.cpu_request)
        self.mem_target = int(self.threshold * self.mem_request)

        self.cpu_weight = cpu_weight
        self.mem_weight = mem_weight

        self.initialize_metrics()

    def initialize_metrics(self) -> None:
        self.metrics = {
            "cpu_usage": 0,
            "mem_usage": 0,
            "traffic_in": 0,
            "traffic_out": 0,
            "latency": 0.0,
        }

    def collect_metrics(self) -> None:
        self.initialize_metrics()
        self._collect_container_metrics()
        self._collect_latency()

    def _collect_container_metrics(self) -> None:
        pods_regex = "|".join(self.pod_names)
        base = f"namespace='{self.namespace}', pod=~'{pods_regex}'"

        queries = {
            "cpu_usage": f"sum(irate(container_cpu_usage_seconds_total{{{base}}}[5m]))",
            "mem_usage": f"sum(container_memory_working_set_bytes{{{base}}})",
            # "traffic_in": f"sum(irate(container_network_receive_bytes_total{{{base}}}[5m]))",
            # "traffic_out": f"sum(irate(container_network_transmit_bytes_total{{{base}}}[5m]))",
        }

        transforms = {
            "cpu_usage": lambda v: int(float(v) * 1000),
            "mem_usage": lambda v: int(float(v) / 1_000_000), # 1_048_576
            # "traffic_in": lambda v: int(float(v) / 1_000), # 1_024
            # "traffic_out": lambda v: int(float(v) / 1_000), # 1_024
        }

        for key, query in queries.items():
            res = self.fetch_prom(query)
            if res:
                self.metrics[key] = transforms[key](res[0]["value"][1])
            else:
                logger.warning("No %s data from Prometheus", key)

    def _collect_latency(self) -> None:
        res = self.fetch_prom(
            "locust_requests_avg_response_time{method='GET', name='/cart'}"
        )
        self.metrics["latency"] = round(float(res[0]["value"][1]), 3) if res else 0.0

    def update_desired_replicas(self) -> None:
        cpu_target_usage = self.num_pods * self.cpu_target
        mem_target_usage = self.num_pods * self.mem_target

        if cpu_target_usage == 0 or mem_target_usage == 0:
            logger.error("Target usage is zero, skipping scaling decision")
            return

        desired_replicas_cpu = math.ceil(self.num_pods * (self.metrics["cpu_usage"] / cpu_target_usage))
        desired_replicas_mem = math.ceil(self.num_pods * (self.metrics["mem_usage"] / mem_target_usage))

        weighted_replicas = (desired_replicas_cpu * self.cpu_weight) + (desired_replicas_mem * self.mem_weight)

        self.desired_replicas = max(self.min_pods, min(math.ceil(weighted_replicas), self.max_pods))
