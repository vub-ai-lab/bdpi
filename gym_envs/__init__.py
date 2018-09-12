from gym.envs.registration import registry, register, make, spec

register(
    id='LargeGrid-v0',
    entry_point='gym_envs.myGrid:myGrid',
    kwargs={'y': 29, 'x': 27}
)

register(
    id='Table-v0',
    entry_point='gym_envs.table:Table',
    kwargs={}
)
