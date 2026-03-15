from gymnasium.envs.registration import register

register(
    id='Redis-v0',
    entry_point='gwydion.envs:Redis',
)