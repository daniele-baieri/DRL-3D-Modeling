from collections import namedtuple

HParams = namedtuple(
    'HParams', [
        'num_prims_side',
        'cuboid_side',
        'movable_vertices_per_prim',
        'possible_slides'
        'r_alpha_1',
        'r_alpha_2',
        'episode_len',
        'learning_rate',
        'replay_buffer_len',
        'batch_size',
        'reward_discount_factor'
    ]
)