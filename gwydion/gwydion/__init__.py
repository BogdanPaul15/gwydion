from gymnasium.envs.registration import register

register(
    id='Redis-v0',
    entry_point='gwydion.envs:Redis',
)

register(
    id='OnlineBoutique-v0',
    entry_point='gwydion.envs:OnlineBoutique',
)
