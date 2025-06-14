"""
This script provides functionality to record and save observations, actions, 
and game states from an Atari environment.
"""
from typing import List
import os
import pickle
import hydra
from omegaconf import DictConfig
import logging

from actions_lists import *
from classes.helper import *
from agents.utils import save_frames_as_mp4
from classes.envs import *
from classes.envs.renderer import get_human_renderer, get_image_renderer
log = logging.getLogger('main')
log.setLevel(logging.INFO)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def make_observations(config: DictConfig, actions: List[str],
                      name: str) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    recorder = AtariEnvRecorder(
        config.recording_with_bb) if config.recording else None
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, config.task, recorder=recorder, renderer=renderer, image_renderer=image_renderer)
    if config.recording:
        recorder.start_recording()
    obs, game_state = env.reset()
    observations = [obs]
    game_states = [game_state]
    ret_actions = []
    all_rewards = 0
    for idx, action in enumerate(actions):
        if action == 'RESTART':
            obs, game_state = env.reset()
        else:
            obs, game_state = env.step(action)

        ret_actions.append(action)
        observations.append(obs)
        game_states.append(game_state)
        
        all_rewards += env.reward
        
        if game_state == GameState.RESTART:
            log.info(f'Restart after {idx + 1} actions')
            if action == 'RESTART':
                log.info("-- HARD RESTART")
        
        # log.info(f'cum wins {env.cum_wins} cum losses {env.cum_losses}')
        
    log.info(f'All rewards: {all_rewards}')
    log.info(f'Num actions: {len(ret_actions)}')
    
    # Save the video
    if config.recording:
        frames = recorder.end_recording()
        save_frames_as_mp4(frames[max(0, config.obs_index):],
                           frameskip=env.frameskip,
                           path='./videos/',
                           filename=f'video_{name}.mp4')

    # Save the observations, actions, and game states
    os.makedirs("saved_data", exist_ok=True)
    with open(f'saved_data/{name}.pickle', "wb") as f:
        pickle.dump((observations[max(0, config.obs_index):],
                     ret_actions[max(0, config.obs_index):],
                     game_states[max(0, config.obs_index):]), f)
    print(f'Saved data to saved_data/{name}.pickle')
    print(f'Num ret_actions: {len(ret_actions[max(0, config.obs_index):])}')


def manual(config: DictConfig) -> None:
    """
    Manually runs an environment to collect observations, actions, and game 
    states, and saves them to a file.
    """
    renderer = get_human_renderer(config)
    env = create_atari_env(config, config.task, renderer=renderer)
    env_player = EnvPlayer(env)
    observations, actions, game_states = env_player.run(
        slow=config.slow_manual_control)
    # save the demonstration
    os.makedirs("saved_data", exist_ok=True)
    save_fn = f'saved_data/obs_manual_{config.task}{config.obs_suffix}.pickle'
    with open(save_fn, "wb") as f:
        pickle.dump((observations, actions, game_states), f)
    print(f'Saved data to {save_fn}')
    print(f'Num actions: {len(actions)}')
    print(actions)


# # Configure logging
# def configure_logging(debug_mode: bool):
#     logging.getLogger().setLevel(logging.DEBUG if debug_mode else
#                                        logging.INFO)
#     logging.getLogger("requests").setLevel(logging.WARNING)
#     logging.getLogger("openai").setLevel(logging.WARNING)
#     logging.debug("Debug mode is enabled")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:

    # configure_logging(config.debug_mode)

    # Initialize game-specific constants
    set_global_constants(config.task)
    
    set_seed(config.seed)
    
    # make_random_observations(config)
    # return

    if config.manual_control:
        manual(config)
    elif config.task == 'MontezumaRevenge':
        
        make_observations(config, montezuma_actions_basic17,
                          f'obs_{config.task}_basic17')
        
    #     from actions_lists.actions_list_extra import montezuma_actions_works_current
    #     make_observations(config, montezuma_actions_works_current,
    #                       f'obs_{config.task}_works_current')
    # elif config.task == 'MontezumaRevengeAlt':
    #     from actions_lists.actions_list_extra import montezuma_alt_actions_works
    #     make_observations(config, montezuma_alt_actions_works,
    #                       f'obs_{config.task}_works')
    elif config.task == 'Pitfall':
        make_observations(config, pitfall_actions_basics1,
                          f'obs_{config.task}_basic1')
    elif config.task == 'Breakout':
        make_observations(config, breakout_actions_basic1,
                          f'obs_{config.task}_basic1')
    elif config.task == 'Pong':
    
        make_observations(config, pong_actions_basic2,
                          f'obs_{config.task}_basic2')
    #     from actions_lists.actions_list_extra import pong_actions_works
    #     make_observations(config, pong_actions_works,
    #                       f'obs_{config.task}_works')
    #     # from actions_lists.actions_list_extra import pong_actions_worldcoder
    #     # make_observations(config, pong_actions_worldcoder,
    #     #                   f'obs_{config.task}_worldcoder')
    # elif config.task == 'PongAlt':
    #     from actions_lists.actions_list_extra import pong_alt_actions_works
    #     make_observations(config, pong_alt_actions_works,
    #                       f'obs_{config.task}_works')
    #     # from actions_lists.actions_list_extra import pong_alt_actions_worldcoder
    #     # make_observations(config, pong_alt_actions_worldcoder,
    #     #                   f'obs_{config.task}_worldcoder')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
