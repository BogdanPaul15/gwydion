from abc import ABC, abstractmethod

import time
import requests

import kubernetes
from kubernetes import client

TOKEN = ""
HOST = ""
PROMETHEUS_URL = ""

class BaseDeploymentWorkload(ABC):
    """Abstract base class for all Kubernetes deployment workloads.

    This class handles K8s API communication, metrics fetching, and state management.

    Attributes:
        k8s (bool): Indicates if running against a real K8s cluster or simulation.
        name (str): The name of the deployment.
        namespace (str): The Kubernetes namespace.
        min_pods (int): Minimum replica boundary.
        max_pods (int): Maximum replica boundary.
        sleep_time (float, optional): Wait time (in seconds) for API retries.
        pod_names (List[str]): List of currently active pod names.
        num_previous_pods (int): Number of pods in the previous observation step.
        num_pods (int): Current number of pods.
        desired_replicas (int): Target number of replicas calculated by the workload.
        metrics (dict): Dictionary storing metrics.
    """
    def __init__(self, k8s, name, namespace, min_pods, max_pods, sleep_time=0.2, **kwargs):
        """Initializes the BaseDeploymentWorkload with deployment configs.
        
        Args:
            k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
            name (str): Name of the deployment workload.
            namespace (str): Namespace name of the deployment workload.
            min_pods (int): Minimum replica count allowed per deployment.
            max_pods (int): Maximum replica count allowed per deployment.
            sleep_time (float): Time (in seconds) between API retries.
        """
        self.k8s = k8s
        self.name = name
        self.namespace = namespace

        self.min_pods = min_pods
        self.max_pods = max_pods

        self.sleep_time = sleep_time

        self.pod_names = []
        self.num_previous_pods = 1
        self.num_pods = 1
        self.desired_replicas = 1

        self.metrics = {}

        if self.k8s:
            self._initialize_k8s_client()
            self._refresh_pods()

    def _initialize_k8s_client(self):
        """Sets up the Kubernetes API client."""
        self.config = client.Configuration()
        self.config.verify_ssl = False
        self.config.api_key = {"authorization": "Bearer " + TOKEN}
        self.config.host = HOST
        self.client = client.ApiClient(self.config)
        self.v1 = client.CoreV1Api(self.client)
        self.apps_v1 = client.AppsV1Api(self.client)

    def _refresh_pods(self):
        """Fetches the latest deployment state and pod names from K8s."""
        self.pod_names = []
        pods = self.v1.list_namespaced_pod(namespace=self.namespace,
                                           label_selector="app=" + self.name)
        for pod in pods.items:
            if pod.metadata.labels["app"] == self.name:
                self.pod_names.append(pod.metadata.name)

        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name,
                                                                         namespace=self.namespace)
        self.num_previous_pods = self.num_pods
        self.num_pods = self.deployment_object.spec.replicas

    @classmethod
    def from_config(cls, cfg: dict, k8s: bool):
        """Factory method to instantiate a workload from a configuration dictionary.

        This method acts as a mapper between the nested YAML configuration structure
        and the class constructor. It flattens the dictionary into specific parameters 
        for pod boundaries, resource requests/limits, and scaling objectives.

        Args:
            cfg (dict): The configuration dictionary for a single deployment.
            k8s (bool): If True, the workload will attempt to connect
                to a live Kubernetes cluster via the API.

        Returns:
            BaseDeploymentWorkload: A fully initialized instance of the workload 
                subclass (e.g., RedisWorkload, OnlineBoutiqueWorkload).
        """
        pods_cfg = cfg["pods"]
        res_cfg = cfg["resources"]
        scaling_cfg = cfg["scaling"]

        return cls(
            k8s=k8s,
            name=cfg["name"],
            namespace=cfg["namespace"],
            min_pods=pods_cfg["min"],
            max_pods=pods_cfg["max"],

            cpu_request=res_cfg["requests"]["cpu"],
            cpu_limit=res_cfg["limits"]["cpu"],
            mem_request=res_cfg["requests"]["mem"],
            mem_limit=res_cfg["limits"]["mem"],

            cpu_weight=scaling_cfg["cpu_weight"],
            mem_weight=scaling_cfg["mem_weight"],
            threshold=scaling_cfg["threshold"],
        )

    def update_obs_k8s(self):
        """The main observation cycle: fetching K8s objects, fetching metrics,
            and calculate desired replicas.
        """
        self._refresh_pods()
        self.collect_metrics()
        self.update_desired_replicas()

    def update_deployment(self, new_replicas):
        """Prepares the deployment object for scaling and triggers the patch request.

        Args:
            new_replicas (int): The target number of pod replicas for current deployment.
        """
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name,
                                                                         namespace=self.namespace)
        self.num_previous_pods = self.deployment_object.spec.replicas
        self.deployment_object.spec.replicas = new_replicas

        self.patch_deployment(new_replicas)

    def patch_deployment(self, new_replicas):
        """Executes the K8s API patch request to update the deployment scale.

        Args:
            new_replicas (int): The target number of pod replicas.
        """
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=self.name, namespace=self.namespace, body=self.deployment_object
            )
        except kubernetes.client.exceptions.ApiException as e:
            print(e)
            print(f"Retrying in {self.sleep_time}s...")
            time.sleep(self.sleep_time)
            return self.update_deployment(new_replicas)

    def fetch_prom(self, query):
        """Queries the Prometheus API and returns resulting data payload.

        Args:
            query (str): The PromQL query string to execute.

        Returns:
            list: The result array from Prometheus JSON response.
        """
        try:
            response = requests.get(
                PROMETHEUS_URL + "/api/v1/query",
                params={"query": query},
                timeout=5
            )
        except requests.exceptions.RequestException as e:
            print(e)
            print(f"Retrying in {self.sleep_time}s...")
            time.sleep(self.sleep_time)
            return self.fetch_prom(query)

        if response.json()["status"] != "success":
            print(f"Error processing the request: {response.json()['status']}")
            print(f"The Error is: {response.json()['error']}")
            print(f"Retrying in {self.sleep_time}s...")
            time.sleep(self.sleep_time)
            return self.fetch_prom(query)

        result = response.json()["data"]["result"]
        return result

    def deploy_pod_replicas(self, n, env):
        """Attempts to scale-out the deployment by `n` pods.
        
        Args:
            n (int): The number of pods to add.
            env (BaseEnv): The Gymnasium environment instance.
        """
        env.none_counter = 0
        replicas = self.num_pods + n

        if replicas <= self.max_pods:
            if self.k8s:
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas

            return

        env.constraint_max_pod_replicas = True

    def terminate_pod_replicas(self, n, env):
        """Attempts to scale-in the deployment by `n` pods.

        Args:
            n (int): The number of pods to remove.
            env (BaseEnv): The Gymnasium environment instance.
        """
        env.none_counter = 0
        replicas = self.num_pods - n

        if replicas >= self.min_pods:
            if self.k8s:
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas

            return

        env.constraint_min_pod_replicas = True

    @abstractmethod
    def collect_metrics(self):
        """Fetches custom metrics to evaluate workload health."""

    @abstractmethod
    def update_desired_replicas(self):
        """Calculates the required number of replicas based on collected metrics."""
