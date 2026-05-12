import csv
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .base import BaseEnv

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

    def get_observation_space(self) -> spaces.Box:
        recommendation = self.deployment_list[ID_RECOMMENDATION]
        productcatalog = self.deployment_list[ID_PRODUCT_CATALOG]
        cartservice = self.deployment_list[ID_CART_SERVICE]
        adservice = self.deployment_list[ID_AD_SERVICE]
        paymentservice = self.deployment_list[ID_PAYMENT_SERVICE]
        shippingservice = self.deployment_list[ID_SHIPPING_SERVICE]
        currencyservice = self.deployment_list[ID_CURRENCY_SERVICE]
        rediscart = self.deployment_list[ID_REDIS_CART]
        checkoutservice = self.deployment_list[ID_CHECKOUT_SERVICE]
        frontend = self.deployment_list[ID_FRONTEND]
        email = self.deployment_list[ID_EMAIL]
        return spaces.Box(
            low=np.array([
                recommendation.min_pods, 0, 0, 0, 0, 0,
                productcatalog.min_pods, 0, 0, 0, 0, 0,
                cartservice.min_pods, 0, 0, 0, 0, 0,
                adservice.min_pods, 0, 0, 0, 0, 0,
                paymentservice.min_pods, 0, 0, 0, 0, 0,
                shippingservice.min_pods, 0, 0, 0, 0, 0,
                currencyservice.min_pods, 0, 0, 0, 0, 0,
                rediscart.min_pods, 0, 0, 0, 0, 0,
                checkoutservice.min_pods, 0, 0, 0, 0, 0,
                frontend.min_pods, 0, 0, 0, 0, 0,
                email.min_pods, 0, 0, 0, 0, 0,
                0,
            ]), high=np.array([
                recommendation.max_pods, recommendation.max_pods, 1000, 1000, 20000, 20000,
                productcatalog.max_pods, productcatalog.max_pods, 1000, 1000, 20000, 20000,
                cartservice.max_pods, cartservice.max_pods, 1000, 1000, 20000, 20000,
                adservice.max_pods, adservice.max_pods, 1000, 1000, 20000, 20000,
                paymentservice.max_pods, paymentservice.max_pods, 1000, 1000, 20000, 20000,
                shippingservice.max_pods, shippingservice.max_pods, 1000, 1000, 20000, 20000,
                currencyservice.max_pods, currencyservice.max_pods, 1000, 1000, 20000, 20000,
                rediscart.max_pods, rediscart.max_pods, 1000, 1000, 20000, 20000,
                checkoutservice.max_pods, checkoutservice.max_pods, 1000, 1000, 20000, 20000,
                frontend.max_pods, frontend.max_pods, 1000, 1000, 20000, 20000,
                email.max_pods, email.max_pods, 1000, 1000, 20000, 20000,
                25,
            ]),
            dtype=np.float32
        )

    def get_state(self) -> np.ndarray:
        recommendation = self.deployment_list[ID_RECOMMENDATION]
        productcatalog = self.deployment_list[ID_PRODUCT_CATALOG]
        cartservice = self.deployment_list[ID_CART_SERVICE]
        adservice = self.deployment_list[ID_AD_SERVICE]
        paymentservice = self.deployment_list[ID_PAYMENT_SERVICE]
        shippingservice = self.deployment_list[ID_SHIPPING_SERVICE]
        currencyservice = self.deployment_list[ID_CURRENCY_SERVICE]
        rediscart = self.deployment_list[ID_REDIS_CART]
        checkoutservice = self.deployment_list[ID_CHECKOUT_SERVICE]
        frontend = self.deployment_list[ID_FRONTEND]
        email = self.deployment_list[ID_EMAIL]
        return np.array([
            recommendation.num_pods, recommendation.desired_replicas,
            recommendation.metrics["cpu_usage"], recommendation.metrics["mem_usage"],
            recommendation.metrics["traffic_in"], recommendation.metrics["traffic_out"],
            productcatalog.num_pods, productcatalog.desired_replicas,
            productcatalog.metrics["cpu_usage"], productcatalog.metrics["mem_usage"],
            productcatalog.metrics["traffic_in"], productcatalog.metrics["traffic_out"],
            cartservice.num_pods, cartservice.desired_replicas,
            cartservice.metrics["cpu_usage"], cartservice.metrics["mem_usage"],
            cartservice.metrics["traffic_in"], cartservice.metrics["traffic_out"],
            adservice.num_pods, adservice.desired_replicas,
            adservice.metrics["cpu_usage"], adservice.metrics["mem_usage"],
            adservice.metrics["traffic_in"], adservice.metrics["traffic_out"],
            paymentservice.num_pods, paymentservice.desired_replicas,
            paymentservice.metrics["cpu_usage"], paymentservice.metrics["mem_usage"],
            paymentservice.metrics["traffic_in"], paymentservice.metrics["traffic_out"],
            shippingservice.num_pods, shippingservice.desired_replicas,
            shippingservice.metrics["cpu_usage"], shippingservice.metrics["mem_usage"],
            shippingservice.metrics["traffic_in"], shippingservice.metrics["traffic_out"],
            currencyservice.num_pods, currencyservice.desired_replicas,
            currencyservice.metrics["cpu_usage"], currencyservice.metrics["mem_usage"],
            currencyservice.metrics["traffic_in"], currencyservice.metrics["traffic_out"],
            rediscart.num_pods, rediscart.desired_replicas,
            rediscart.metrics["cpu_usage"], rediscart.metrics["mem_usage"],
            rediscart.metrics["traffic_in"], rediscart.metrics["traffic_out"],
            checkoutservice.num_pods, checkoutservice.desired_replicas,
            checkoutservice.metrics["cpu_usage"], checkoutservice.metrics["mem_usage"],
            checkoutservice.metrics["traffic_in"], checkoutservice.metrics["traffic_out"],
            frontend.num_pods, frontend.desired_replicas,
            frontend.metrics["cpu_usage"], frontend.metrics["mem_usage"],
            frontend.metrics["traffic_in"], frontend.metrics["traffic_out"],
            email.num_pods, email.desired_replicas,
            email.metrics["cpu_usage"], email.metrics["mem_usage"],
            email.metrics["traffic_in"], email.metrics["traffic_out"],
            self.none_counter,
        ], dtype=np.float32)

    def collect_obs(self, obs, date, latency):
        row_data = {
            "date": date,
            "recommendationservice_latency": float(f"{latency:.3f}")
        }

        for i, d in enumerate(self.deployment_list):
            idx = i * 6
            row_data.update({
                f"{d.name}_num_pods": int(obs[idx]),
                f"{d.name}_desired_replicas": int(obs[idx + 1]),
                f"{d.name}_cpu_usage": int(obs[idx + 2]),
                f"{d.name}_mem_usage": int(obs[idx + 3]),
                f"{d.name}_traffic_in": int(obs[idx + 4]),
                f"{d.name}_traffic_out": int(obs[idx + 5])
            })
        self.episode_buffer.append(row_data)

    def save_obs_to_csv(self):
        if not self.episode_buffer:
            return

        obs_file = self.obs_file
        file_exists = Path(obs_file).exists()

        with open(obs_file, "a+", encoding="utf-8", newline="") as f:
            fields = ["date"]
            for d in self.deployment_list:
                fields.extend([
                    f"{d.name}_num_pods",
                    f"{d.name}_desired_replicas",
                    f"{d.name}_cpu_usage",
                    f"{d.name}_mem_usage",
                    f"{d.name}_traffic_in",
                    f"{d.name}_traffic_out"
                ])
            fields.append("recommendationservice_latency")

            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()

            writer.writerows(self.episode_buffer)

        self.episode_buffer = []
