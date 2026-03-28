from abc import ABC, abstractmethod
from typing import Self
import logging

import requests

import kubernetes
from kubernetes import client

from gwydion.envs.util import backoff

TOKEN = ""
HOST = ""
PROMETHEUS_URL = ""

logger = logging.getLogger(__name__)

class Deployment(ABC):
    """Abstract base class for Kubernetes deployments.

    This class handles K8s API communication, metrics fetching, and state management.

    Attributes:
        k8s (bool): Indicates if running against a real K8s cluster or simulation.
        name (str): The name of the deployment.
        namespace (str): The Kubernetes namespace.
        min_pods (int): Minimum replica boundary.
        max_pods (int): Maximum replica boundary.
        pod_names (List[str]): List of currently active pod names.
        num_previous_pods (int): Number of pods in the previous observation step.
        num_pods (int): Current number of pods.
        desired_replicas (int): Target number of replicas calculated by the deployment.
        metrics (dict): Dictionary storing metrics.
    """
    def __init__(self, k8s: bool, name: str, namespace: str, min_pods: int,
                  max_pods: int, **kwargs):
        """Initializes the Deployment with deployment config.
        
        Args:
            k8s (bool): If True, interacts with a real K8s cluster. If False, runs simulation.
            name (str): Name of the deployment.
            namespace (str): Namespace name of the deployment.
            min_pods (int): Minimum replica count allowed for deployment.
            max_pods (int): Maximum replica count allowed for deployment.
        """
        self.k8s = k8s
        self.name = name
        self.namespace = namespace

        self.min_pods = min_pods
        self.max_pods = max_pods

        self.pod_names = []
        self.num_previous_pods = 1
        self.num_pods = 1
        self.desired_replicas = 1

        self.metrics = {}

        if self.k8s:
            self.deployment_object = None
            self._initialize_k8s_client()
            self._refresh_pods()

    def _initialize_k8s_client(self) -> None:
        """Sets up the Kubernetes API client."""
        self.config = client.Configuration()
        self.config.verify_ssl = False
        self.config.api_key = {"authorization": "Bearer " + TOKEN}
        self.config.host = HOST
        self.client = client.ApiClient(self.config)
        self.v1 = client.CoreV1Api(self.client)
        self.apps_v1 = client.AppsV1Api(self.client)

    def _refresh_pods(self) -> None:
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

        logger.debug("Refreshed %s | Pods: %d -> %d", self.name, self.num_previous_pods,
                      self.num_pods)

    @classmethod
    def from_config(cls, cfg: dict, k8s: bool) -> Self:
        """Factory method to instantiate a deployment from a configuration dictionary.

        This method acts as a mapper between the nested YAML configuration structure
        and the class constructor. It flattens the dictionary into specific parameters 
        for pod boundaries, resource requests/limits, and scaling objectives.

        Args:
            cfg (dict): The configuration dictionary for a single deployment.
            k8s (bool): If True, the deployment will attempt to connect
                to a live Kubernetes cluster via the API.

        Returns:
            Deployment: A fully initialized instance of the deployment 
                subclass (e.g., RedisDeployment, OnlineBoutiqueDeployment).
        """
        pods_cfg = cfg["pods"]
        resources_cfg = cfg["resources"]
        scaling_cfg = cfg["scaling"]

        return cls(
            k8s=k8s,
            name=cfg["name"],
            namespace=cfg["namespace"],
            min_pods=pods_cfg["min"],
            max_pods=pods_cfg["max"],

            cpu_request=resources_cfg["requests"]["cpu"],
            cpu_limit=resources_cfg["limits"]["cpu"],
            mem_request=resources_cfg["requests"]["mem"],
            mem_limit=resources_cfg["limits"]["mem"],

            cpu_weight=scaling_cfg["cpu_weight"],
            mem_weight=scaling_cfg["mem_weight"],
            threshold=scaling_cfg["threshold"],
        )

    def update_obs_k8s(self) -> None:
        """The main observation cycle: fetching K8s objects, collecting metrics,
            and calculate desired replicas.
        """
        self._refresh_pods()
        self.collect_metrics()
        self.update_desired_replicas()

    def update_deployment(self, new_replicas: int) -> None:
        """Prepares the deployment object for scaling and triggers the patch request.

        Args:
            new_replicas (int): The target number of pod replicas for current deployment.
        """
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name,
                                                                         namespace=self.namespace)
        self.num_previous_pods = self.deployment_object.spec.replicas
        self.deployment_object.spec.replicas = new_replicas

        self.patch_deployment()

    @backoff(delay=0.5, retries=3, exceptions=(kubernetes.client.exceptions.ApiException,))
    def patch_deployment(self) -> None:
        """Executes the K8s API patch request to update the deployment scale.
        
        Raises:
            kubernetes.client.exceptions.ApiException: If the patch request fails after
              all retries.
        """
        logger.debug("Patching %s | Current replicas: %d | Target replicas: %d",
                      self.name, self.num_previous_pods, self.deployment_object.spec.replicas)
        self.apps_v1.patch_namespaced_deployment(
            name=self.name, namespace=self.namespace, body=self.deployment_object
        )
        logger.debug("Patch successful for %s", self.name)

    @backoff(delay=0.5, retries=3, exceptions=(requests.exceptions.RequestException,))
    def fetch_prom(self, query: str) -> list:
        """Queries the Prometheus API and returns resulting data payload.

        Args:
            query (str): The PromQL query string to execute.

        Returns:
            list: The result array from Prometheus JSON response.

        Raises:
            requests.exceptions.RequestException: If the request fails after all retries.
            RuntimeError: If Prometheus returns a non-success status after retries.
        """
        response = requests.get(
            PROMETHEUS_URL + "/api/v1/query",
            params={"query": query},
            timeout=5
        )

        if response.json()["status"] != "success":
            logger.error("Prometheus query failed for %s: %s", self.name, 
                         response.json().get("error", ""))
            raise RuntimeError(f"Prometheus error: {response.json()['status']} \
                               - {response.json().get('error', '')}")
        return response.json()["data"]["result"]

    def deploy_pod_replicas(self, num_replicas: int) -> bool:
        """Attempts to scale-out the deployment by `num_replicas` pods.

        Args:
            num_replicas (int): The number of pods to add.

        Returns:
            bool: True if the operation would exceed the maximum allowed pods (max_pods),
                  False if scaling was successful and within limits.
        """
        new_total = self.num_pods + num_replicas

        if new_total <= self.max_pods:
            if self.k8s:
                self.update_deployment(new_total)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = new_total
            return False
        return True

    def terminate_pod_replicas(self, num_replicas: int) -> bool:
        """Attempts to scale-in the deployment by `num_replicas` pods.

        Args:
            num_replicas (int): The number of pods to remove.
        
        Returns:
            bool: True if the operation would fall below the minimum allowed pods (min_pods),
                  False if scaling was successful and within limits.
        """
        new_total = self.num_pods - num_replicas

        if new_total >= self.min_pods:
            if self.k8s:
                self.update_deployment(new_total)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = new_total
            return False
        return True

    @abstractmethod
    def initialize_metrics(self) -> None:
        """Initializes the specific metrics required for Deployment state."""
        raise NotImplementedError

    @abstractmethod
    def collect_metrics(self) -> None:
        """Fetches custom metrics to evaluate deployment health."""
        raise NotImplementedError

    @abstractmethod
    def update_desired_replicas(self) -> None:
        """Calculates the required number of replicas based on collected metrics."""
        raise NotImplementedError
