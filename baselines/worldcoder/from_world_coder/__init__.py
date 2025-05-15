import numpy as np
# from .synthesizer import synthesis
from .llm_utils import LLM

def refine_world_model(
    init_transit_code,
    init_reward_code,
    experiences,
    llm_default_args = {'model': 'gpt-4o', 'temperature': 1.0,},
    max_budget=100, # $100
    bandits_C=20.0,
    np_rng=None,
):
    assert len(experiences) > 0
    assert isinstance(np_rng, np.random.Generator)
    print(f'refine_world_model with {len(experiences)} experiences')

    llm = LLM(seed=0, default_args=llm_default_args)

    best_output, cache_dir = synthesis(
        experiences,
        init_transit_code=init_transit_code,
        init_reward_code=init_reward_code,
        llm=llm,
        np_rng=np_rng,
        max_budget=max_budget,
        bandits_C=bandits_C,
    )
    transit_code = best_output['transit_code']
    reward_code = best_output['reward_code']
    fitness = best_output['success_ratio']
    return fitness, (transit_code, reward_code, best_output, cache_dir)
