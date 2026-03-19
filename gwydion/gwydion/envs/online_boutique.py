import csv

import numpy as np
from gymnasium import spaces
from gwydion.envs import base
from gwydion.envs.deployment import get_online_boutique_deployment_list

MAX_STEPS = 25 # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

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
    def __init__(self, k8s=False, reward_strategy=None, waiting_period=5):
        super().__init__(
            name="online_boutique_gym",
            num_apps=11,
            deployments=["recommendationservice", "productcatalogservice", "cartservice",
                        "adservice", "paymentservice", "shippingservice", "currencyservice",
                        "redis-cart", "checkoutservice", "frontend", "emailservice"],
            k8s=k8s,
            reward_strategy=reward_strategy,
            waiting_period=waiting_period
        )

        self.deployment_list = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        self.observation_space = self.get_observation_space()

        # TODO remove this
        self.file_results = "results.csv"

        if not k8s:
            self.load_dataset()
            self.traffic = self.simulation_traffic("frontend")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.deployment_list = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return self.get_state(), self.info

    def take_action(self, action, id):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            print('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            self.none_counter += 1
            print("[Take Action] SELECTED ACTION: DO NOTHING ...")

        elif action == ACTION_ADD_1_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deployment_list[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            print("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deployment_list[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deployment_list[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            print("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deployment_list[id].terminate_pod_replicas(7, self)

        else:
            print('[Take Action] Unrecognized Action: ' + str(action))

    def get_observation_space(self):
        return spaces.Box(
            low=np.array([
                self.min_pods,  # Number of Pods  -- 1) recommendationservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 2) productcatalogservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 3) cartservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 4) adservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 5) paymentservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 6) shippingservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 7) currencyservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 8) redis-cart
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 9) checkoutservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 10) frontend
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                self.min_pods,  # Number of Pods -- 11) emailservice
                0,  # CPU Usage (in m)
                0,  # MEM Usage (in MiB)
                0,  # None Counter
            ]), high=np.array([
                self.max_pods,  # Number of Pods -- 1)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 2)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 3)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 4)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 5)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 6)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 7)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 8)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 9)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 10)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                self.max_pods,  # Number of Pods -- 11)
                1000,  # CPU Usage (in m)
                1000,  # MEM Usage (in MiB)
                10,      # None counter
            ]),
            dtype=np.float32
        )

    def get_state(self):
        ob = (
<<<<<<< HEAD
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
=======
            self.deploymentList[ID_recommendation].num_pods,
            self.deploymentList[ID_recommendation].metrics["cpu_usage"],
            self.deploymentList[ID_recommendation].metrics["mem_usage"],
            self.deploymentList[ID_product_catalog].num_pods,
            self.deploymentList[ID_product_catalog].metrics["cpu_usage"],
            self.deploymentList[ID_product_catalog].metrics["mem_usage"],
            self.deploymentList[ID_cart_service].num_pods,
            self.deploymentList[ID_cart_service].metrics["cpu_usage"],
            self.deploymentList[ID_cart_service].metrics["mem_usage"],
            self.deploymentList[ID_ad_service].num_pods,
            self.deploymentList[ID_ad_service].metrics["cpu_usage"],
            self.deploymentList[ID_ad_service].metrics["mem_usage"],
            self.deploymentList[ID_payment_service].num_pods,
            self.deploymentList[ID_payment_service].metrics["cpu_usage"],
            self.deploymentList[ID_payment_service].metrics["mem_usage"],
            self.deploymentList[ID_shipping_service].num_pods,
            self.deploymentList[ID_shipping_service].metrics["cpu_usage"],
            self.deploymentList[ID_shipping_service].metrics["mem_usage"],
            self.deploymentList[ID_currency_service].num_pods,
            self.deploymentList[ID_currency_service].metrics["cpu_usage"],
            self.deploymentList[ID_currency_service].metrics["mem_usage"],
            self.deploymentList[ID_redis_cart].num_pods,
            self.deploymentList[ID_redis_cart].metrics["cpu_usage"],
            self.deploymentList[ID_redis_cart].metrics["mem_usage"],
            self.deploymentList[ID_checkout_service].num_pods,
            self.deploymentList[ID_checkout_service].metrics["cpu_usage"],
            self.deploymentList[ID_checkout_service].metrics["mem_usage"],
            self.deploymentList[ID_frontend].num_pods,
            self.deploymentList[ID_frontend].metrics["cpu_usage"],
            self.deploymentList[ID_frontend].metrics["mem_usage"],
            self.deploymentList[ID_email].num_pods,
            self.deploymentList[ID_email].metrics["cpu_usage"],
            self.deploymentList[ID_email].metrics["mem_usage"],
>>>>>>> 6da6a5e (refactor: modify environments to accept new deployment workload)
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
