import logging
import argparse
import time

from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO, MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gwydion.envs import Redis, OnlineBoutique
from gwydion.rewards import CostStrategy, LatencyStrategy
from gwydion.utils import test_model

# Logging

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='ppo', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "onlineboutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')
parser.add_argument('--seed', default=None, type=int, help='Random seed for reproducibility')

parser.add_argument('--training', default=True, action="store_true", help='Training mode')
parser.add_argument('--testing', default=False, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path',
                    default='logs/a2c_env_onlineboutique_goal_cost_k8s_False_totalSteps_500000/a2c_env_redis_goal_cost_k8s_False_totalSteps_500000.zip',
                    help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path',
                    default='logs/a2c_env_onlineboutique_goal_latency_k8s_False_totalSteps_500000/a2c_env_onlineboutique_goal_latency_k8s_False_totalSteps_500000.zip',
                    help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=50000, help='The steps for saving.')
parser.add_argument('--total_steps', default=250000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log, seed=None):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500, seed=seed)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=seed)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, seed=seed)  # , n_steps=steps
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, goal, seed=None):
    def make_env():
        if use_case == "redis":
            if goal == "cost":
                return Redis(config_path="configs/redis.yaml", reward_strategy=CostStrategy(), seed=seed)
            else:
                return Redis(config_path="configs/redis.yaml", reward_strategy=LatencyStrategy(target_id=0, threshold=250.0), seed=seed)
        elif use_case == 'onlineboutique':
            if goal == "cost":
                return OnlineBoutique(config_path="configs/online_boutique.yaml", reward_strategy=CostStrategy(), seed=seed)
            else:
                return OnlineBoutique(config_path="configs/online_boutique.yaml", reward_strategy=LatencyStrategy(target_id=9, threshold=3000.0), seed=seed)
        else:
            raise ValueError(f"Unknown use_case: {use_case}")

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return env


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal
    seed = args.seed
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, goal, seed=seed)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "results/" + use_case + "/" + scenario + "/" + goal + "/"

    name = alg + "_env_" + use_case + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        # if loading:  # resume training
        #     model = get_load_model(alg, tensorboard_log, load_path)
        #     model.set_env(env)
        #     model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        # else:
        model = get_model(alg, env, tensorboard_log, seed=seed)
        model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        # model.save(name)

    # if testing:
    #     model = get_load_model(alg, tensorboard_log, test_path)
    #     test_model(model, env, n_episodes=25, n_steps=25, smoothing_window=5, fig_name=name + "_check2.png")


if __name__ == "__main__":
    main()
