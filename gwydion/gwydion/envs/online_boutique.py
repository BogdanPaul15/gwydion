import csv
import os

import numpy as np
from gymnasium import spaces
from gwydion.envs import BaseEnv

ID_RECOMMENDATION = 0
ID_PRODUCT_CATALOG = 1
ID_CART_SERVICE = 2
ID_AD_SERVICE = 3
ID_PAYMENT_SERVICE = 4
ID_SHIPPING_SERVICE = 5
ID_CURRENCY_SERVICE = 6
ID_REDIS_CART = 7
ID_CHECKOUT_SERVICE = 8
ID_FRONTEND = 9
ID_EMAIL = 10

class OnlineBoutique(BaseEnv):
    """Horizontal Scaling for Online Boutique in K8s - an Gymnasium gym environment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = self.get_observation_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        return self.get_state(), self.info

    def get_observation_space(self):
        return spaces.Box(
            low=np.array([
                self.deployment_list[ID_RECOMMENDATION].min_pods,  # Number of Pods  -- 1) recommendationservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_PRODUCT_CATALOG].min_pods,  # Number of Pods -- 2) productcatalogservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_CART_SERVICE].min_pods,  # Number of Pods -- 3) cartservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_AD_SERVICE].min_pods,  # Number of Pods -- 4) adservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_PAYMENT_SERVICE].min_pods,  # Number of Pods -- 5) paymentservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_SHIPPING_SERVICE].min_pods,  # Number of Pods -- 6) shippingservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_CURRENCY_SERVICE].min_pods,  # Number of Pods -- 7) currencyservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_REDIS_CART].min_pods,  # Number of Pods -- 8) redis-cart
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_CHECKOUT_SERVICE].min_pods,  # Number of Pods -- 9) checkoutservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_FRONTEND].min_pods,  # Number of Pods -- 10) frontend
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[ID_EMAIL].min_pods,  # Number of Pods -- 11) emailservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                0,  # None Counter
            ]), high=np.array([
                self.deployment_list[ID_RECOMMENDATION].max_pods,  # Number of Pods -- 1)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_PRODUCT_CATALOG].max_pods,  # Number of Pods -- 2)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_CART_SERVICE].max_pods,  # Number of Pods -- 3)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_AD_SERVICE].max_pods,  # Number of Pods -- 4)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_PAYMENT_SERVICE].max_pods,  # Number of Pods -- 5)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_SHIPPING_SERVICE].max_pods,  # Number of Pods -- 6)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_CURRENCY_SERVICE].max_pods,  # Number of Pods -- 7)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_REDIS_CART].max_pods,  # Number of Pods -- 8)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_CHECKOUT_SERVICE].max_pods,  # Number of Pods -- 9)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_FRONTEND].max_pods,  # Number of Pods -- 10)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[ID_EMAIL].max_pods,  # Number of Pods -- 11)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                10,      # None counter
            ]),
            dtype=np.float32
        )

    def get_state(self):
        ob = (
            self.deployment_list[ID_RECOMMENDATION].num_pods,
            self.deployment_list[ID_RECOMMENDATION].metrics["cpu_usage"],
            self.deployment_list[ID_RECOMMENDATION].metrics["mem_usage"],
            self.deployment_list[ID_PRODUCT_CATALOG].num_pods,
            self.deployment_list[ID_PRODUCT_CATALOG].metrics["cpu_usage"],
            self.deployment_list[ID_PRODUCT_CATALOG].metrics["mem_usage"],
            self.deployment_list[ID_CART_SERVICE].num_pods,
            self.deployment_list[ID_CART_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_CART_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_AD_SERVICE].num_pods,
            self.deployment_list[ID_AD_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_AD_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_PAYMENT_SERVICE].num_pods,
            self.deployment_list[ID_PAYMENT_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_PAYMENT_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_SHIPPING_SERVICE].num_pods,
            self.deployment_list[ID_SHIPPING_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_SHIPPING_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_CURRENCY_SERVICE].num_pods,
            self.deployment_list[ID_CURRENCY_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_CURRENCY_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_REDIS_CART].num_pods,
            self.deployment_list[ID_REDIS_CART].metrics["cpu_usage"],
            self.deployment_list[ID_REDIS_CART].metrics["mem_usage"],
            self.deployment_list[ID_CHECKOUT_SERVICE].num_pods,
            self.deployment_list[ID_CHECKOUT_SERVICE].metrics["cpu_usage"],
            self.deployment_list[ID_CHECKOUT_SERVICE].metrics["mem_usage"],
            self.deployment_list[ID_FRONTEND].num_pods,
            self.deployment_list[ID_FRONTEND].metrics["cpu_usage"],
            self.deployment_list[ID_FRONTEND].metrics["mem_usage"],
            self.deployment_list[ID_EMAIL].num_pods,
            self.deployment_list[ID_EMAIL].metrics["cpu_usage"],
            self.deployment_list[ID_EMAIL].metrics["mem_usage"],
            self.none_counter,
        )

        # return self.normalize(ob)
        return ob

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file_exists = os.path.isfile(obs_file)

        with open(obs_file, "a+", encoding="utf-8", newline="") as f:
            fields = ["date"]
            for d in self.deployment_list:
                fields.extend([
                    f"{d.name}_num_pods",
                    f"{d.name}_cpu_usage",
                    f"{d.name}_mem_usage",
                ])
            fields.append("recommendationservice_latency")

            writer = csv.DictWriter(f, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            row_data = {
                "date": date,
                "recommendationservice_latency": float(f"{latency:.3f}")
            }

            for i, d in enumerate(self.deployment_list):
                idx = i * 3
                row_data[f"{d.name}_num_pods"] = int(f"{obs[idx]}")
                row_data[f"{d.name}_cpu_usage"] = int(f"{obs[idx + 1]}")
                row_data[f"{d.name}_mem_usage"] = int(f"{obs[idx + 2]}")

            writer.writerow(row_data)
