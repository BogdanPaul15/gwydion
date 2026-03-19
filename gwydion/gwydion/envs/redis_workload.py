import math

from gwydion.envs import workload

class RedisWorkload(workload.BaseDeploymentWorkload):
    """Concrete workload implementation for Redis gym environment.
    
    Scales based on a weighted CPU and MEM usage, Network I/O,
    and by tracking Redis specific latency.
    """
    def __init__(self, k8s, name, namespace, min_pods, max_pods,
                 cpu_request, cpu_limit, mem_request, mem_limit,
                 cpu_weight=0.7, mem_weight=0.3, threshold=0.75, sleep_time=0.2):
        super().__init__(k8s, name, namespace, min_pods, max_pods, sleep_time)

        self.cpu_request = cpu_request
        self.mem_request = mem_request
        self.threshold = threshold

        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit

        self.cpu_target = int(self.threshold * self.cpu_request)
        self.mem_target = int(self.threshold * self.mem_request)

        self.cpu_weight = cpu_weight
        self.mem_weight = mem_weight

    def collect_metrics(self):
        self.metrics["cpu_usage"] = 0
        self.metrics["mem_usage"] = 0
        self.metrics["received_traffic"] = 0
        self.metrics["transmit_traffic"] = 0

        # TODO: maybe this part can be aggregated into one query for each metric
        for pod in self.pod_names:
            query_cpu = f"sum(irate(container_cpu_usage_seconds_total{{namespace='{self.namespace}', pod='{pod}'}}[5m]))"
            query_mem = f"sum(irate(container_memory_working_set_bytes{{namespace='{self.namespace}', pod='{pod}'}}[5m]))"
            query_rec = f"sum(irate(container_network_receive_bytes_total{{namespace='{self.namespace}', pod='{pod}'}}[5m]))"
            query_trans = f"sum(irate(container_network_transmit_bytes_total{{namespace='{self.namespace}', pod='{pod}'}}[5m]))"

            res_cpu = self.fetch_prom(query_cpu)
            if res_cpu:
                self.metrics["cpu_usage"] += int(float(res_cpu[0]["value"][1]) * 1000)

            res_mem = self.fetch_prom(query_mem)
            if res_mem:
                self.metrics["mem_usage"] += int(float(res_mem[0]["value"][1]) / 1000000)

            res_rec = self.fetch_prom(query_rec)
            if res_rec:
                self.metrics["received_traffic"] += int(float(res_rec[0]["value"][1]) / 1000)

            res_trans = self.fetch_prom(query_trans)
            if res_trans:
                self.metrics["transmit_traffic"] += int(float(res_trans[0]["value"][1]) / 1000)

        # TODO: should not be hardcoded
        if self.name == "redis-leader":
            query_dur = "sum(irate(redis_commands_duration_seconds_total[5m]))"
            query_proc = "sum(irate(redis_commands_processed_total[5m]))"

            redis_duration = 0
            redis_processed = 0

            res_dur = self.fetch_prom(query_dur)
            if res_dur:
                redis_duration = float(res_dur[0]["value"][1]) * 1000

            res_proc = self.fetch_prom(query_proc)
            if res_proc:
                redis_processed = float(res_proc[0]["value"][1])

            if redis_processed != 0:
                self.metrics["latency"] = float(f"{redis_duration / redis_processed:.3f}")
            else:
                self.metrics["latency"] = float(f"{redis_duration:.3f}")

    def update_desired_replicas(self):
        cpu_target_usage = self.num_pods * self.cpu_target
        mem_target_usage = self.num_pods * self.mem_target

        desired_replicas_cpu = math.ceil(self.num_pods * (self.metrics["cpu_usage"] / cpu_target_usage))
        desired_replicas_mem = math.ceil(self.num_pods * (self.metrics["mem_usage"] / mem_target_usage))

        weighted_replicas = (desired_replicas_cpu * self.cpu_weight) + (desired_replicas_mem * self.mem_weight)

        self.desired_replicas = max(self.min_pods, min(math.ceil(weighted_replicas), self.max_pods))
