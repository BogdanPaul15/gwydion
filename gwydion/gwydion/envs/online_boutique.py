import csv
from pathlib import Path

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
                recommendation.min_pods, 0, 0,
                productcatalog.min_pods, 0, 0,
                cartservice.min_pods, 0, 0,
                adservice.min_pods, 0, 0,
                paymentservice.min_pods, 0, 0,
                shippingservice.min_pods, 0, 0,
                currencyservice.min_pods, 0, 0,
                rediscart.min_pods, 0, 0,
                checkoutservice.min_pods, 0, 0,
                frontend.min_pods, 0, 0,
                email.min_pods, 0, 0,
                0,
            ]), high=np.array([
                recommendation.max_pods, 1000, 1000,
                productcatalog.max_pods, 1000, 1000,
                cartservice.max_pods, 1000, 1000,
                adservice.max_pods, 1000, 1000,
                paymentservice.max_pods, 1000, 1000,
                shippingservice.max_pods, 1000, 1000,
                currencyservice.max_pods, 1000, 1000,
                rediscart.max_pods, 1000, 1000,
                checkoutservice.max_pods, 1000, 1000,
                frontend.max_pods, 1000, 1000,
                email.max_pods, 1000, 1000,
                10,
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
            recommendation.num_pods, recommendation.metrics["cpu_usage"], recommendation.metrics["mem_usage"],
            productcatalog.num_pods, productcatalog.metrics["cpu_usage"], productcatalog.metrics["mem_usage"],
            cartservice.num_pods, cartservice.metrics["cpu_usage"], cartservice.metrics["mem_usage"],
            adservice.num_pods, adservice.metrics["cpu_usage"], adservice.metrics["mem_usage"],
            paymentservice.num_pods, paymentservice.metrics["cpu_usage"], paymentservice.metrics["mem_usage"],
            shippingservice.num_pods, shippingservice.metrics["cpu_usage"], shippingservice.metrics["mem_usage"],
            currencyservice.num_pods, currencyservice.metrics["cpu_usage"], currencyservice.metrics["mem_usage"],
            rediscart.num_pods, rediscart.metrics["cpu_usage"], rediscart.metrics["mem_usage"],
            checkoutservice.num_pods, checkoutservice.metrics["cpu_usage"], checkoutservice.metrics["mem_usage"],
            frontend.num_pods, frontend.metrics["cpu_usage"], frontend.metrics["mem_usage"],
            email.num_pods, email.metrics["cpu_usage"], email.metrics["mem_usage"],
            self.none_counter,
        ], dtype=np.float32)

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file_exists = Path(obs_file).exists()

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
                row_data[f"{d.name}_num_pods"] = int(obs[idx])
                row_data[f"{d.name}_cpu_usage"] = int(obs[idx + 1])
                row_data[f"{d.name}_mem_usage"] = int(obs[idx + 2])

            writer.writerow(row_data)
