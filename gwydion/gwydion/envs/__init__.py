from .deployment_registry import register, build_deployment_list
# from gwydion.gwydion.envs.old_redis import OldRedis
# from gwydion.gwydion.envs.old_online_boutique import OnlineBoutique
from .base import BaseEnv
from .redis import Redis
from .online_boutique import OnlineBoutique
from .redis_deployment import RedisDeployment
from .online_boutique_deployment import OnlineBoutiqueDeployment
