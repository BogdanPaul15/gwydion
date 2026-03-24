import csv

import numpy as np
from gymnasium import spaces
from gwydion.envs import base

ID_recommendation = 0
ID_product_catalog = 1
ID_cart_service = 2
ID_ad_service = 3
ID_payment_service = 4
ID_shipping_service = 5
ID_currency_service = 6
ID_redis_cart = 7
ID_checkout_service = 8
ID_frontend = 9
ID_email = 10

class OnlineBoutique(base.BaseEnv):
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
                self.deployment_list[0].min_pods,  # Number of Pods  -- 1) recommendationservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[1].min_pods,  # Number of Pods -- 2) productcatalogservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[2].min_pods,  # Number of Pods -- 3) cartservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[3].min_pods,  # Number of Pods -- 4) adservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[4].min_pods,  # Number of Pods -- 5) paymentservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[5].min_pods,  # Number of Pods -- 6) shippingservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[6].min_pods,  # Number of Pods -- 7) currencyservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[7].min_pods,  # Number of Pods -- 8) redis-cart
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[8].min_pods,  # Number of Pods -- 9) checkoutservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[9].min_pods,  # Number of Pods -- 10) frontend
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.deployment_list[10].min_pods,  # Number of Pods -- 11) emailservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                0,  # None Counter
            ]), high=np.array([
                self.deployment_list[0].max_pods,  # Number of Pods -- 1)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[1].max_pods,  # Number of Pods -- 2)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[2].max_pods,  # Number of Pods -- 3)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[3].max_pods,  # Number of Pods -- 4)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[4].max_pods,  # Number of Pods -- 5)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[5].max_pods,  # Number of Pods -- 6)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[6].max_pods,  # Number of Pods -- 7)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[7].max_pods,  # Number of Pods -- 8)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[8].max_pods,  # Number of Pods -- 9)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[9].max_pods,  # Number of Pods -- 10)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.deployment_list[10].max_pods,  # Number of Pods -- 11)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                10,      # None counter
            ]),
            dtype=np.float32
        )

    def get_state(self):
        ob = (
            self.deployment_list[ID_recommendation].num_pods,
            self.deployment_list[ID_recommendation].metrics["cpu_usage"],
            self.deployment_list[ID_recommendation].metrics["mem_usage"],
            self.deployment_list[ID_product_catalog].num_pods,
            self.deployment_list[ID_product_catalog].metrics["cpu_usage"],
            self.deployment_list[ID_product_catalog].metrics["mem_usage"],
            self.deployment_list[ID_cart_service].num_pods,
            self.deployment_list[ID_cart_service].metrics["cpu_usage"],
            self.deployment_list[ID_cart_service].metrics["mem_usage"],
            self.deployment_list[ID_ad_service].num_pods,
            self.deployment_list[ID_ad_service].metrics["cpu_usage"],
            self.deployment_list[ID_ad_service].metrics["mem_usage"],
            self.deployment_list[ID_payment_service].num_pods,
            self.deployment_list[ID_payment_service].metrics["cpu_usage"],
            self.deployment_list[ID_payment_service].metrics["mem_usage"],
            self.deployment_list[ID_shipping_service].num_pods,
            self.deployment_list[ID_shipping_service].metrics["cpu_usage"],
            self.deployment_list[ID_shipping_service].metrics["mem_usage"],
            self.deployment_list[ID_currency_service].num_pods,
            self.deployment_list[ID_currency_service].metrics["cpu_usage"],
            self.deployment_list[ID_currency_service].metrics["mem_usage"],
            self.deployment_list[ID_redis_cart].num_pods,
            self.deployment_list[ID_redis_cart].metrics["cpu_usage"],
            self.deployment_list[ID_redis_cart].metrics["mem_usage"],
            self.deployment_list[ID_checkout_service].num_pods,
            self.deployment_list[ID_checkout_service].metrics["cpu_usage"],
            self.deployment_list[ID_checkout_service].metrics["mem_usage"],
            self.deployment_list[ID_frontend].num_pods,
            self.deployment_list[ID_frontend].metrics["cpu_usage"],
            self.deployment_list[ID_frontend].metrics["mem_usage"],
            self.deployment_list[ID_email].num_pods,
            self.deployment_list[ID_email].metrics["cpu_usage"],
            self.deployment_list[ID_email].metrics["mem_usage"],
            self.none_counter,
        )

        # return self.normalize(ob)
        return ob

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file = open(obs_file, 'a+', encoding='utf-8', newline='')
        fields = []
        with file:
            fields.append('date')
            for d in self.deployment_list:
                fields.append(d.name + '_num_pods')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_mem_usage')

            writer = csv.DictWriter(file, fieldnames=fields)
            # TODO this writes an independent header for each row
            # writer.writeheader()
            writer.writerow(
                {'date': date,
                 'recommendationservice_num_pods': int(f"{obs[0]}"),
                 'recommendationservice_cpu_usage': int(f"{obs[1]}"),
                 'recommendationservice_mem_usage': int(f"{obs[2]}"),
                 'recommendationservice_latency': float(f"{latency:.3f}"),
                 'productcatalogservice_num_pods': int(f"{obs[3]}"),
                 'productcatalogservice_cpu_usage': int(f"{obs[4]}"),
                 'productcatalogservice_mem_usage': int(f"{obs[5]}"),
                 'cartservice_num_pods': int(f"{obs[6]}"),
                 'cartservice_cpu_usage': int(f"{obs[7]}"),
                 'cartservice_mem_usage': int(f"{obs[8]}"),
                 'adservice_num_pods': int(f"{obs[9]}"),
                 'adservice_cpu_usage': int(f"{obs[10]}"),
                 'adservice_mem_usage': int(f"{obs[11]}"),
                 'paymentservice_num_pods': int(f"{obs[12]}"),
                 'paymentservice_cpu_usage': int(f"{obs[13]}"),
                 'paymentservice_mem_usage': int(f"{obs[14]}"),
                 'shippingservice_num_pods': int(f"{obs[15]}"),
                 'shippingservice_cpu_usage': int(f"{obs[16]}"),
                 'shippingservice_mem_usage': int(f"{obs[17]}"),
                 'currencyservice_num_pods': int(f"{obs[18]}"),
                 'currencyservice_cpu_usage': int(f"{obs[19]}"),
                 'currencyservice_mem_usage': int(f"{obs[20]}"),
                 'redis-cart_num_pods': int(f"{obs[21]}"),
                 'redis-cart_cpu_usage': int(f"{obs[22]}"),
                 'redis-cart_mem_usage': int(f"{obs[23]}"),
                 'checkoutservice_num_pods': int(f"{obs[24]}"),
                 'checkoutservice_cpu_usage': int(f"{obs[25]}"),
                 'checkoutservice_mem_usage': int(f"{obs[26]}"),
                 'frontend_num_pods': int(f"{obs[27]}"),
                 'frontend_cpu_usage': int(f"{obs[28]}"),
                 'frontend_mem_usage': int(f"{obs[29]}"),
                 'emailservice_num_pods': int(f"{obs[30]}"),
                 'emailservice_cpu_usage': int(f"{obs[31]}"),
                 'emailservice_mem_usage': int(f"{obs[32]}"),
                 }
            )
