import logging
from openai_hf_interface import create_llm
import asyncio

from classes.envs.renderer import get_human_renderer
from data.atari import load_atari_observations
from prompts.synthesizer import danger_att_prompt, explain_event_symmetric_prompt

log = logging.getLogger('main')
log.setLevel(logging.DEBUG)

import hydra
from omegaconf import OmegaConf
from pathlib import Path

from classes.helper import set_global_constants
from openai_hf_interface import choose_provider
import numpy as np
import random


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def setup(config):
    cache_mode = 'disk_to_memory' if config.use_memory else 'disk'
    llm = create_llm('gpt-4o-2024-08-06')
    llm.setup_cache(cache_mode, database_path=config.database_path)
    llm.set_default_kwargs({'timeout': 60})
    return llm


def test(config):
    llm = setup(config)

    actions = [
        'RIGHT', 'NOOP', 'LEFTFIRE', 'LEFTFIRE', "RIGHTFIRE", "DOWN",
        'RIGHTFIRE', 'RIGHT', 'RIGHT'
    ]

    funcs = [
        """\
def alter_player_objects(obj_list: ObjList, action: str, touch_side=3, touch_percent=1.0) -> ObjList:
    if action == 'RIGHT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        conveyer_belts = obj_list.get_objs_by_obj_type('conveyer_belt')
        for player_obj in player_objs:
            for conveyer_belt in conveyer_belts:
                if player_obj.touches(conveyer_belt, touch_side, touch_percent):
                    player_obj.velocity_x = RandomValues([2])
                    break
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.2) -> ObjList:
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for conveyer_belt_obj in conveyer_belt_objs:  # conveyer_belt_obj is of type Obj
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):  # check if player_obj touches conveyer_belt_obj
                # If action is 'NOOP', set the bottom_side of player_obj to the top_side of conveyer_belt_obj
                player_obj.bottom_side = RandomValues([conveyer_belt_obj.top_side])
                break  # Avoid setting the attribute more than once for each player object
    return obj_list""", """\
def alter_player_objects(obj_list: ObjList, action: str, touch_side=3, touch_percent=1.0) -> ObjList:
    if action == 'LEFTFIRE':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, touch_side, touch_percent):
                    player_obj.velocity_x = RandomValues([-2])
                    break
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str, touch_side=3, touch_percent=1.0) -> ObjList:
    if action == 'LEFTFIRE':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, touch_side, touch_percent):
                    player_obj.velocity_y = RandomValues([-6])
                    break
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == 'RIGHTFIRE':
        player_objs = obj_list.get_objs_by_obj_type('player')
        for player_obj in player_objs:
            if player_obj.velocity_y == 0:
                player_obj.velocity_y = RandomValues([-6])
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == 'DOWN':
        player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
        platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
        
        for player_obj in player_objs:  # player_obj is of type Obj
            for platform_obj in platform_objs:  # platform_obj is of type Obj
                if player_obj.touches(platform_obj, touch_side, touch_percent):
                    # Set the player's center_y to the platform's center_y using RandomValues
                    player_obj.center_y = RandomValues([platform_obj.center_y])
                    break  # Avoid setting the attribute more than once
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == 'RIGHTFIRE':
        player_objs = obj_list.get_objs_by_obj_type('player')
        for player_obj in player_objs:
            if player_obj.velocity_x == -4:
                player_obj.velocity_x = RandomValues([-5])
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == 'RIGHT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        conveyer_belts = obj_list.get_objs_by_obj_type('conveyer_belt')
        for player_obj in player_objs:
            for conveyer_belt in conveyer_belts:
                if player_obj.touches(conveyer_belt, touch_side, touch_percent):
                    player_obj.velocity_x = RandomValues([2])
                    break
    return obj_list
""", """\
def alter_player_objects(obj_list: ObjList, action: str) -> ObjList:
    if action == 'RIGHT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        conveyer_belts = obj_list.get_objs_by_obj_type('conveyer_belt')
        for player_obj in player_objs:
            for conveyer_belt in conveyer_belts:
                if player_obj.touches(conveyer_belt, touch_side, touch_percent):
                    player_obj.velocity_y = RandomValues([2])
                    break
    return obj_list
"""
    ]

    outputs = llm.prompt([
        explain_event_symmetric_prompt.format(
            obj_type='player', action=action, func=func)
        for action, func in zip(actions, funcs)
    ],
                         temperature=0,
                         seed=config.seed)
    for x in outputs:
        print(x)


def quick():
    from agents.agent import read_world_model_from_path, save_world_model_to_path
    world_model = read_world_model_from_path(
        'tmp_world_models/0f303b5e-7ef1-4e7b-b23c-7efb2f1ebe34.pickle')
    no_callables_world_model = world_model.remove_callables()
    save_world_model_to_path(no_callables_world_model)


def pomdp_synth(config):
    import asyncio
    from learners.synthesizer import MultiTimestepActionSynthesizer
    from classes.helper import StateTransitionTriplet

    llm = setup(config)
    synthesizer = MultiTimestepActionSynthesizer(config, 'player', llm)

    observations, actions, game_states = load_atari_observations(
        config.task + config.obs_suffix)
    c = []
    for i in range(len(actions)):
        x = StateTransitionTriplet(observations[i],
                                   actions[i],
                                   observations[i + 1],
                                   input_game_state=game_states[i],
                                   output_game_state=game_states[i + 1])
        c.append(x)
        log.info(i)
        log.info(x)
    log.info(c[:17][-1].input_state)
    res = asyncio.run(synthesizer.a_synthesize(c[:17]))
    log.info(res[0])


def pomdp2_synth(config):
    import asyncio
    from learners.synthesizer import MultiTimestepStatusChangeSynthesizer
    from classes.helper import StateTransitionTriplet

    llm = setup(config)
    synthesizer = MultiTimestepStatusChangeSynthesizer(config, 'beam', llm)

    observations, actions, game_states = load_atari_observations(
        config.task + config.obs_suffix)

    for idx, obs in enumerate(observations):
        print(f'Observation {idx}:')
        print([obj.str_w_id() for obj in obs.objs if obj.obj_type == 'beam'])
        print()

    c = []
    for i in range(len(actions)):
        x = StateTransitionTriplet(observations[i],
                                   actions[i],
                                   observations[i + 1],
                                   input_game_state=game_states[i],
                                   output_game_state=game_states[i + 1])
        c.append(x)
    # 43, 55
    log.info(c[:43][-1])
    res = asyncio.run(synthesizer.a_synthesize(c[:43]))
    log.info(res[0])


def pomdp2_synth_2(config):
    import asyncio
    from learners.synthesizer import PassiveCreationSynthesizer
    from classes.helper import StateTransitionTriplet

    llm = setup(config)
    synthesizer = PassiveCreationSynthesizer(config, 'beam', llm)

    observations, actions, game_states = load_atari_observations(
        config.task + config.obs_suffix)

    for idx, obs in enumerate(observations):
        print(f'Observation {idx}:')
        print([obj.str_w_id() for obj in obs.objs if obj.obj_type == 'beam'])
        print()

    c = []
    for i in range(len(actions)):
        x = StateTransitionTriplet(observations[i],
                                   actions[i],
                                   observations[i + 1],
                                   input_game_state=game_states[i],
                                   output_game_state=game_states[i + 1])
        c.append(x)
    # 43, 55
    log.info(c[:13][-1])
    res = asyncio.run(synthesizer.a_synthesize(c[:13]))
    log.info(res[0])


def player_history_txt(obj):
    return f"""\
- The player object has 
    history['velocity_x'] = {obj.history['velocity_x'][-10:]}
    history['velocity_y'] = {obj.history['velocity_y'][-10:]}
    history['n_touch_above'] = {obj.history['n_touch_above'][-10:]}
    history['n_touch_below'] = {obj.history['n_touch_below'][-10:]}
    history['n_touch_left'] = {obj.history['n_touch_left'][-10:]}
    history['n_touch_right'] = {obj.history['n_touch_right'][-10:]}
"""


def test_dead(config):
    from classes.helper import StateTransitionTriplet

    observations, actions, game_states = load_atari_observations(
        config.task + config.obs_suffix)
    c = []
    for i in range(len(actions)):
        x = StateTransitionTriplet(observations[i],
                                   actions[i],
                                   observations[i + 1],
                                   input_game_state=game_states[i],
                                   output_game_state=game_states[i + 1])
        c.append(x)

    from learners.synthesizer import RestartSynthesizer
    llm = setup(config)
    synthesizer = RestartSynthesizer(config, 'player', llm)

    for idx, x in enumerate(c):
        if x.output_game_state == 'RESTART':
            log.info(idx)
            res = asyncio.run(synthesizer.a_synthesize(c[:idx + 1]))
            log.info('output:')
            for idx, output in enumerate(res):
                log.info(idx)
                log.info(output)
            breakpoint()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Configure OpenAI/HuggingFace API
    choose_provider(config.provider)

    # Set random seeds
    set_seed(config.seed)

    # Initialize game-specific constants
    set_global_constants(config.task)

    # test(config)
    # quick()
    # pomdp_synth(config)
    # pomdp2_synth(config)
    # pomdp2_synth(config)
    test_dead(config)


if __name__ == '__main__':
    main()

# import cProfile
# import pstats
# from classes.generic import StateTransitionTriplet
# c = [StateTransitionTriplet(*data) for data in zip(observations[:3], actions, observations[1:3], game_states, game_states[1:])]
# log.info(len(c))
# # fast update of world model
# # Profile the method call
# profiler = cProfile.Profile()
# profiler.enable()  # Start profiling
# learner.quick_fix_world_model(c)  # Call the function
# profiler.disable()  # Stop profiling

# # Print or save profiling results
# with open('profile_results2.txt', 'w') as f:
#     stats = pstats.Stats(profiler, stream=f)
#     stats.strip_dirs().sort_stats('cumulative').print_stats()
