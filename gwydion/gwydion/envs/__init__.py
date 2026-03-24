from .workload_registry import register, build_deployment_list
# from gwydion.gwydion.envs.old_redis import OldRedis
# from gwydion.gwydion.envs.old_online_boutique import OnlineBoutique
from .base import BaseEnv
from .redis import Redis
from .online_boutique import OnlineBoutique
from .redis_workload import RedisWorkload
from .online_boutique_workload import OnlineBoutiqueWorkload
