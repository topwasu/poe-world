"""
This script provides functionality to record and save observations, actions, 
and game states from an Atari environment.
"""
from typing import List
import os
import pickle
import hydra
from omegaconf import DictConfig
from PIL import Image
import logging

from actions_lists import *
from classes.helper import *
from agents.utils import save_frames_as_mp4
from classes.envs import *
from classes.envs.renderer import get_human_renderer, get_image_renderer
log = logging.getLogger('main')
log.setLevel(logging.INFO)


def make_montezuma_fig1(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['NOOP', 'DOWN']
    
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, 'MontezumaRevenge', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
    
    frames = recorder.end_recording()
    os.makedirs('./images', exist_ok=True)
    for idx in range(len(frames)):
        # Save the frame as an image
        frame_image = Image.fromarray(frames[idx])
        width, height = frame_image.size
        cropped_frame_image = frame_image.crop((0, height * 0.22, width, height * 0.87))  # Crop 10% from top and bottom
        cropped_frame_image.save(f'./images/MontezumaRevenge_fig1_frame_{idx}.png')
        

def make_montezuma_dead(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFTFIRE', 'LEFTFIRE', 'LEFTFIRE', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP']
    
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, 'MontezumaRevenge', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
    
    frames = recorder.end_recording()
    os.makedirs('./images', exist_ok=True)
    for idx in range(len(frames)-10, len(frames)):
        # Save the frame as an image
        frame_image = Image.fromarray(frames[idx])
        width, height = frame_image.size
        cropped_frame_image = frame_image.crop((0, height * 0.22, width, height * 0.87))  # Crop 10% from top and bottom
        cropped_frame_image.save(f'./images/MontezumaRevenge_dead_frame_{idx}.png')

def make_montezuma_success(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFTFIRE', 'LEFTFIRE', 'LEFTFIRE', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UPLEFTFIRE', 'LEFTFIRE', 'LEFTFIRE', 'NOOP', 'NOOP']
    
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, 'MontezumaRevenge', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
    
    frames = recorder.end_recording()
    os.makedirs('./images', exist_ok=True)
    for idx in range(len(frames)-10, len(frames)):
        # Save the frame as an image
        frame_image = Image.fromarray(frames[idx])
        width, height = frame_image.size
        cropped_frame_image = frame_image.crop((0, height * 0.22, width, height * 0.87))  # Crop 10% from top and bottom
        cropped_frame_image.save(f'./images/MontezumaRevenge_success_frame_{idx}.png')


def make_montezuma_alt_gameplay(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP']
    
    
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, 'MontezumaRevengeAlt', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
            
    frames = recorder.end_recording()
    os.makedirs('./images', exist_ok=True)
    for idx in range(len(frames)):
        if idx % 5 == 1:
            # Save the frame as an image
            frame_image = Image.fromarray(frames[idx])
            width, height = frame_image.size
            cropped_frame_image = frame_image.crop((0, height * 0.03, width, height * 0.89))  # Crop 10% from top and bottom
            cropped_frame_image.save(f'./images/MRAlt_frame_{idx}.png')

def make_pong_alt_gameplay(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RESTART', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP']
    
    recorder = AtariEnvRecorder(
        config.recording_with_bb)
    renderer = get_human_renderer(config)
    image_renderer = get_image_renderer(config)
    env = create_atari_env(config, 'PongAlt', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
    
    frames = recorder.end_recording()
    os.makedirs('./images', exist_ok=True)
    for idx in range(len(frames)):
        if idx % 5 == 1:
            # Save the frame as an image
            frame_image = Image.fromarray(frames[idx])
            width, height = frame_image.size
            cropped_frame_image = frame_image.crop((0, height * 0.14, width, height * 0.95))  # Crop 10% from top and bottom
            cropped_frame_image.save(f'./images/PongAlt_frame_{idx}.png')
            
            
def make_pong_planning(config: DictConfig) -> None:
    """
    Executes a series of actions in an Atari environment, records observations
    and game states, and optionally saves a video of the gameplay. The 
    observations, actions, and game states are saved to a pickle file.
    """
    actions = ['NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'RIGHT', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'RIGHT', 'RIGHT', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'LEFT'] + ['NOOP'] * 20 
    actions_list = [
        actions,
        actions[:68] + ['NOOP'] * 20,
        actions[:68] + ['LEFT'] * 3 + ['NOOP'] * 20,
        actions[:73] + ['LEFT'] * 3 + ['NOOP'] * 20,
        actions[:73] + ['RIGHT'] * 3 + ['NOOP'] * 20,
    ]
    
    for run_idx, actions in enumerate(actions_list):
    
        recorder = AtariEnvRecorder(
            config.recording_with_bb)
        renderer = get_human_renderer(config)
        image_renderer = get_image_renderer(config)
        env = create_atari_env(config, 'Pong', recorder=recorder, renderer=renderer, image_renderer=image_renderer)
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
        
        frames = recorder.end_recording()
        os.makedirs('./images', exist_ok=True)
        for idx in [68, 73, 78]:
            # Save the frame as an image
            frame_image = Image.fromarray(frames[idx])
            width, height = frame_image.size
            cropped_frame_image = frame_image.crop((0, height * 0.15, width, height * 0.93))  # Crop 10% from top and bottom
            cropped_frame_image.save(f'./images/Pong_planning_run{run_idx}_frame_{idx}.png')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:

    # configure_logging(config.debug_mode)

    # Initialize game-specific constants
    set_global_constants(config.task)
    
    
    # make_pong_planning(config)
    # make_montezuma_fig1(config)
    # make_montezuma_dead(config)
    # make_montezuma_success(config)
    # make_montezuma_alt_gameplay(config)
    make_pong_alt_gameplay(config)
    


if __name__ == '__main__':
    main()
