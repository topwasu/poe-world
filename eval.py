import numpy as np
import logging
from data.atari import load_atari_observations
from classes.helper import StateTransitionTriplet, are_two_obj_lists_equal
from actions_lists import *
from classes.helper import *
from classes.envs import *
from classes.envs.renderer import get_human_renderer, get_image_renderer
from omegaconf import DictConfig

log = logging.getLogger('main')


def obj_lists_partial_match_score(predicted_obj_list, actual_obj_list, mode='partial'):
    moving_obj_types = ['player', 'enemy', 'skull', 'ball']
    predicted_obj_list = ObjList([obj for obj in predicted_obj_list.objs if obj.obj_type in moving_obj_types], no_copy=True)
    actual_obj_list = ObjList([obj for obj in actual_obj_list.objs if obj.obj_type in moving_obj_types], no_copy=True)
    
    if len(predicted_obj_list) != len(actual_obj_list):
        return 0
    
    # Check same number of objects for each type
    
    by_obj_type1 = {}
    for obj in predicted_obj_list.objs:
        if obj.obj_type not in by_obj_type1:
            by_obj_type1[obj.obj_type] = []
        by_obj_type1[obj.obj_type].append(obj)
    
    by_obj_type2 = {}
    for obj in actual_obj_list.objs:
        if obj.obj_type not in by_obj_type2:
            by_obj_type2[obj.obj_type] = []
        by_obj_type2[obj.obj_type].append(obj)
        
    for obj_type in by_obj_type1:
        if obj_type not in by_obj_type2:
            return 0
        if len(by_obj_type1[obj_type]) != len(by_obj_type2[obj_type]):
            return 0
    
    hsh = {}
    sm = 0
    for obj in actual_obj_list:
        best_match_idx = None
        best_match_dist = 1000000
        for idx, obj2 in enumerate(predicted_obj_list):
            if obj.obj_type != obj2.obj_type or idx in hsh:
                continue
            dist = abs(obj.x - obj2.x) + abs(obj.y - obj2.y)
            if dist < best_match_dist:
                best_match_dist = dist
                best_match_idx = idx
        if mode == 'direction':
            dir_v_x = obj.velocity_x
            pred_v_x = predicted_obj_list[best_match_idx].velocity_x
            x_dir_score = 1 if (dir_v_x > 0 and pred_v_x > 0) or (dir_v_x < 0 and pred_v_x < 0) or (dir_v_x == 0 and pred_v_x == 0) else 0
            
            dir_v_y = obj.velocity_y
            pred_v_y = predicted_obj_list[best_match_idx].velocity_y
            y_dir_score = 1 if (dir_v_y > 0 and pred_v_y > 0) or (dir_v_y < 0 and pred_v_y < 0) or (dir_v_y == 0 and pred_v_y == 0) else 0
            score = (x_dir_score + y_dir_score) / 2
        else:
            score = (int(obj.x == predicted_obj_list[best_match_idx].x) + int(obj.y == predicted_obj_list[best_match_idx].y)) / 2
        sm += score
        hsh[best_match_idx] = True
    
    return sm / len(actual_obj_list)


def make_random_observations(config: DictConfig, length) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, config.task, recorder=recorder, renderer=renderer, image_renderer=image_renderer, skip_gameover_if_possible=False)
    if config.recording:
        recorder.start_recording()
    obs, game_state = env.reset()
    observations = [obs]
    game_states = [game_state]
    ret_actions = []
    all_rewards = 0
    # for idx, action in enumerate(actions):
    if length is None:
        n_iterations = 27000
    else:
        n_iterations = length
    for i in range(n_iterations):
        
        if i % 1000 == 0:
            log.info(f'i: {i}')
        
        if config.task.startswith('Pong'):
            action = np.random.choice(['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
        elif config.task.startswith('MontezumaRevenge'):
            action = np.random.choice(['NOOP', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'FIRE', 'LEFTFIRE', 'RIGHTFIRE'])
        
        # logging.debug(f'Action {idx}: {action}')
        if action == 'RESTART':
            obs, game_state = env.reset()
        else:
            obs, game_state = env.step(action)

        ret_actions.append(action)
        observations.append(obs)
        game_states.append(game_state)
        
        all_rewards += env.reward
        
        if game_state == 'GAMEOVER':
            if length is None:
                break
            else:
                obs, game_state = env.reset()

                ret_actions.append(action)
                observations.append(obs)
                game_states.append(game_state)
        # log.info(f'cum wins {env.cum_wins} cum losses {env.cum_losses}')
        
    log.info(all_rewards)
    log.info(f'Num actions: {len(ret_actions)}')
    
    return observations, ret_actions, game_states



def grab_transitions(config):
    # Load observations -- use the same observation for both non-prime and prime versions
    if config.eval.set == 'random':
        observations, actions, game_states = make_random_observations(config, 1000)
    else:
        observations, actions, game_states = load_atari_observations(
            config.task.replace('Alt', '') + config.obs_suffix)
    # observations, actions, game_states = load_atari_observations('manual_MontezumaRevenge_basic')

    # Optional: Use subset of observations
    if config.obs_index != -1:
        observations = observations[config.obs_index:config.obs_index +
                                    config.obs_index_length + 1]
        actions = actions[config.obs_index:config.obs_index +
                            config.obs_index_length]
        game_states = game_states[config.obs_index:config.obs_index +
                                    config.obs_index_length + 1]
        
    transitions = []
    for i in range(len(actions)):
        transitions.append(StateTransitionTriplet(observations[i],
                                                    actions[i],
                                                    observations[i + 1],
                                                    input_game_state=game_states[i],
                                                    output_game_state=game_states[i + 1],
                                                    add_ghost=False))
    return transitions

def evaluate_world_model(config, world_model):
    transitions = grab_transitions(config)
    correct = 0
    for idx, transition in enumerate(transitions):
        # log.info(f'Evaluating transition {idx} of {len(transitions)}')
        if config.method == 'worldcoder':
            predicted_state = world_model.sample_next_scene(transition.input_state, transition.event)
            if config.eval.mode in ['partial', 'direction']:
                correct += obj_lists_partial_match_score(predicted_state, transition.output_state, mode=config.eval.mode)
            else:
                if are_two_obj_lists_equal(predicted_state, transition.output_state):
                    correct += 1
        else:
            memory = transition.input_state.memory
            predicted_state = world_model.sample_next_scene(transition.input_state, transition.event, memory=memory, det=config.det_world_model)
            if config.eval.mode in ['partial', 'direction']:
                correct += obj_lists_partial_match_score(predicted_state, transition.output_state, mode=config.eval.mode)
            else:
                if are_two_obj_lists_equal(predicted_state, transition.output_state):
                    correct += 1
            # log.info(f'transition.input_state[0]: {transition.input_state[0]}')
            # log.info(f'transition.event: {transition.event}')
            # log.info(f'transition.output_state[0]: {transition.output_state[0]}')
            # logprobs = world_model.evaluate_logprobs(transition.input_state, 
            #                                          transition.event, 
            #                                          transition.output_state, 
            #                                          memory=memory)
            # correct += np.exp(logprobs)
        # log.info(f'Accuracy: {correct / (idx + 1)}')
    log.info(f'Final Accuracy on {config.method} {config.task} {config.eval.mode}: {correct / len(transitions)}')
    