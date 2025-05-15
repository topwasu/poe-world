"""Example command:
- Base training:
python run_rl.py task=Pong
python run_rl.py task=MontezumaRevenge

- Finetuning
python run_rl.py task=PongAlt seed=0 \
    pretrained_model_file=baseline_checkpoints/Pong-s0-fs4-pe_positional/best_model.zip

To monitor with Tensorboard:
tensorboard --logdir baseline_logs --port 9000 --host 0.0.0.0
"""

import sys
from os import path
import hydra
import dill as pickle
import logging
import numpy as np
import os
import glob
import logging
import random
from omegaconf import DictConfig
from pathlib import Path
from typing import Callable
from rtpt import RTPT

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                EveryNTimesteps, BaseCallback,
                                                CallbackList, EvalCallback)

from ocatari.environments import PositionHistoryEnv
from ocatari.utils import parser
from baselines.worldcoder import WorldCoder
from classes.envs import *
from classes.envs.object_tracker import *
from classes.envs.renderer import get_human_renderer, get_empty_renderer
from classes.game_utils import *
from classes.helper import set_global_constants, StateTransitionTriplet
from data.atari import load_atari_observations
from learners.world_model_learner import PoEWorldLearner, WorldModelLearner


# Training configuration constants
TRAINING_TIMESTEPS = 20_000_000
CHECKPOINT_FREQUENCY = 100_000
EVAL_FREQUENCY = 100_000
N_EVAL_ENVS = 4
N_EVAL_EPISODES = 4

# Model hyperparameters
POLICY_STR = "MlpPolicy"
ADAM_STEP_SIZE = 0.00025
CLIPPING_EPS = 0.1
N_STEPS = 128
N_EPOCHS = 3
# BATCH_SIZE = 32 * 8  # moved to be defined later based on n_cpus
GAMMA = 0.99
GAE_LAMBDA = 0.95
VF_COEF = 1
ENT_COEF = 0.01

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def get_world_model_learner(config: DictConfig) -> WorldModelLearner:
    """
    Choose the world model learner based on the configuration.
    
    Args:
        config: Hydra configuration object containing all parameters
    
    Returns:
        WorldModelLearner: The world model learner object
    """
    if config.method == 'poe':
        return PoEWorldLearner(config)
    elif config.method == 'worldcoder':
        return WorldCoder(config)
    else:
        raise NotImplementedError
    
    
def get_reward_fn(config: DictConfig) -> Callable:
    if config.task.startswith('Montezuma'):
        def reward_fn(obj_list) -> float:
            try:
                player_obj = obj_list.get_objs_by_obj_type('player')[0]
                key_obj = obj_list.get_objs_by_obj_type('key')[0]
                return 100 if player_obj.overlaps(key_obj) else 0
            except:
                return 0
        return reward_fn
    elif config.task.startswith('Pong'):
        def reward_fn(obj_list) -> float:
            try:
                reward = 0
                player_obj = obj_list.get_objs_by_obj_type('player')[0]
                ball_objs = obj_list.get_objs_by_obj_type('ball')
                for ball_obj in ball_objs:
                    if player_obj.overlaps(ball_obj):
                        reward += 1
                return reward
            except:
                return 0
        return reward_fn
    else:
        raise NotImplementedError


def create_world_as_env(config, rank = 0) -> 'WrappedOCAtariEnv':
    observations, actions, game_states = load_atari_observations(
        config.task.replace('Alt', '') + config.obs_suffix)
    
    transitions = []
    for i in range(len(actions)):
        transitions.append(
            StateTransitionTriplet(observations[i],
                                   actions[i],
                                   observations[i + 1],
                                   input_game_state=game_states[i],
                                   output_game_state=game_states[i + 1]))

    learner = get_world_model_learner(config)
    world_model = learner.synthesize_world_model(transitions)
    world_model.clear_cache()
    
    reward_fn = get_reward_fn(config)
    
    object_tracker = ObjectTracker()
    renderer = get_empty_renderer(config)
    imagine_env = ImaginedAtariEnv(config, world_model, object_tracker,
                                    renderer, seed_frames=observations, reward_fn=reward_fn, rank=rank)
    
    return imagine_env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Create a linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def add_gym_wrapper(config: DictConfig, atari_env: WrappedOCAtariEnv,
                    truncate_length: int = 36000) -> GymAtariEnv:
    """Create a Gym environment wrapper for the Atari game."""
    if config.task in ['MontezumaRevenge', 'MontezumaRevengeAlt']:
        actions = MONTEZUMA_REVENGE_ACTIONS
    elif config.task in ['Pong', 'PongAlt']:
        actions = PONG_ACTIONS
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")
    
    env = GymAtariEnv(atari_env, actions, config, truncate_length=truncate_length)
    
    # Add episode info to the environment
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    return env


def create_env(config: DictConfig, rank: int, seed: int = 0, 
               eval_env: bool=False) -> Callable:
    """Create a vectorized environment with the specified configuration.
    Args:
        eval_env (bool): Whether to create an evaluation environment, this would
            affect the truncate length.
    """

    def _init() -> gym.Env:
        # Set global constants in each subprocess
        set_global_constants(config.task)
        
        if config.use_world_as_env:
            env = create_world_as_env(config, rank=rank)
        else:
            env = create_wrapped_ocatari_env(config, config.task, no_visual=True, skip_gameover_if_possible=False)
        
        if config.use_world_as_env:
            truncate_length = 1000
        elif eval_env and config.task == 'MontezumaRevenge':
            truncate_length = 3000
        else:
            truncate_length = 36000 # Consider lowering this for World Environments
            
        env = add_gym_wrapper(config, env, truncate_length=truncate_length)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def get_latest_checkpoint_by_steps(checkpoint_dir: str) -> str:
    """Find the checkpoint with the highest step count in the given directory."""
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "model_*_steps.zip"))

    if not checkpoint_files:
        return None

    def get_step_number(filepath: str) -> int:
        filename = os.path.basename(filepath)
        return int(filename.split("_")[1])

    checkpoint_files.sort(key=get_step_number)
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint with highest step count: {latest_checkpoint}")
    return latest_checkpoint


class RTPTCallback(BaseCallback):
    """Custom callback for RTPT progress tracking."""

    def __init__(self, total_timesteps: int, check_freq: int = 1024):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.rtpt = RTPT(name_initials='RL',
                         experiment_name='Training',
                         max_iterations=total_timesteps // check_freq)
        self.rtpt.start()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            log.info(f"Step {self.num_timesteps} of {self.total_timesteps}")
            self.rtpt.step()
            eta = self.rtpt._get_eta_str()
            log.info(f"Estimated time remaining: {eta}")
        return True


class RolloutInfoCallback(BaseCallback):
    """Custom callback for tracking rollout information."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_rewards = []
        self.rollout_lengths = []
        self.rollout_episodes = 0
        
    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get('infos', [])
        if not infos:
            return True
            
        for info in infos:
            if 'episode' in info:
                episode_info = info['episode']
                reward = episode_info.get('r', 0)
                length = episode_info.get('l', 0)
                
                self.rollout_rewards.append(reward)
                self.rollout_lengths.append(length)
                self.rollout_episodes += 1
                
                # Log to Tensorboard
                self.logger.record('rollout/ep_rew_mean', np.mean(self.rollout_rewards))
                self.logger.record('rollout/ep_len_mean', np.mean(self.rollout_lengths))
                self.logger.record('rollout/ep_rew_std', np.std(self.rollout_rewards))
                self.logger.record('rollout/ep_len_std', np.std(self.rollout_lengths))
                self.logger.record('rollout/episodes', self.rollout_episodes)
                self.logger.dump(self.num_timesteps)
                
        return True


def setup_callbacks(n_envs: int, eval_env: gym.Env,
                    ckpt_path: Path, config=None) -> CallbackList:
    """Set up evaluation and checkpoint callbacks."""
    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=N_EVAL_EPISODES,
                                 best_model_save_path=str(ckpt_path),
                                 log_path=str(ckpt_path),
                                 eval_freq=max(EVAL_FREQUENCY // n_envs, 1),
                                 deterministic=True,
                                 render=False)

    checkpoint_callback = CheckpointCallback(save_freq=max(
        CHECKPOINT_FREQUENCY // n_envs, 1),
                                             save_path=str(ckpt_path),
                                             name_prefix="model",
                                             save_replay_buffer=True,
                                             save_vecnormalize=False)

    rtpt_callback = RTPTCallback(total_timesteps=TRAINING_TIMESTEPS)
    callbacks = [checkpoint_callback, eval_callback, rtpt_callback]
    if config is not None and config.use_rollout_callback:
        rollout_callback = RolloutInfoCallback()
        callbacks.append(rollout_callback)
    return CallbackList(callbacks)


def create_model(env: gym.Env, config, latest_ckpt: str = None, load_ckpt: str = None) -> PPO:
    """Create or load a PPO model."""
    if latest_ckpt:
        return PPO.load(latest_ckpt, env=env)
    elif load_ckpt:
        # Used when finetuning a pretrained model on alt envs.
        model = PPO.load(load_ckpt, env=env)
        
        if config.ppo.reset_timesteps:
            # Update parameters
            model.learning_rate = linear_schedule(ADAM_STEP_SIZE)
            model.clip_range=linear_schedule(CLIPPING_EPS)
            
            # Reset the num_timesteps counter to 0 for proper tracking during fine-tuning
            model.num_timesteps = 0
            model._episode_num = 0
            
        return model
        
    BATCH_SIZE = 32 * config.n_cpu

    # Set up policy_kwargs for neural net architecture
    policy_kwargs = dict(net_arch=dict(pi=config.ppo.policy_net_arch,
                                       vf=config.ppo.policy_net_arch))

    return PPO(POLICY_STR,
               n_steps=config.ppo.n_steps,
               learning_rate=linear_schedule(ADAM_STEP_SIZE),
               n_epochs=N_EPOCHS,
               batch_size=BATCH_SIZE,
               gamma=GAMMA,
               gae_lambda=GAE_LAMBDA,
               clip_range=linear_schedule(CLIPPING_EPS),
               vf_coef=VF_COEF,
               ent_coef=config.ppo.ent_coef,
               env=env,
               verbose=1,
               seed=config.seed,
               policy_kwargs=policy_kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """Main training loop for the RL agent."""
    set_global_constants(config.task)
    
    set_seed(config.seed)

    # Setup paths and experiment name
    exp_name = f"{config.task}-s{config.seed}-fs{config.n_stacking_frames}-pe_{config.encoding_option}-no_wh"
    if config.use_world_as_env:
        exp_name = 'world-' + exp_name
    if config.pretrained_model_file:
        exp_name = 'finetune-' + exp_name
    if config.ppo.n_steps != N_STEPS:
        exp_name =  exp_name + f'-ns{config.ppo.n_steps}'
    if config.ppo.ent_coef != ENT_COEF:
        exp_name =  exp_name + f'-ec{config.ppo.ent_coef}'
    if config.ppo.policy_net_arch != [64, 64]:
        exp_name = exp_name + f"-arch{config.ppo.policy_net_arch}"

    log_path = Path("baseline_logs", exp_name)
    ckpt_path = Path("baseline_checkpoints", exp_name)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    log.info('Before creating vectorized environments')
    # Create vectorized environments
    eval_env_seed = (config.seed + 42) * 2
    vec_eval_env = SubprocVecEnv([
        create_env(config, rank=i, seed=eval_env_seed, eval_env=True)
        for i in range(N_EVAL_ENVS)
    ],
                             start_method="spawn") 
    vec_env = SubprocVecEnv([
        create_env(config, rank=i, seed=config.seed)
        for i in range(config.n_cpu)
    ],
                        start_method="spawn")
    if config.n_stacking_frames > 1:
        # Frame-stacking with 4 frames
        log.debug(f"Stacking {config.n_stacking_frames} frames")
        vec_eval_env = VecFrameStack(vec_eval_env, config.n_stacking_frames)
        vec_env = VecFrameStack(vec_env, config.n_stacking_frames)

    # Setup callbacks and logger
    log.info('Before creating callbacks')
    callbacks = setup_callbacks(config.n_cpu, vec_eval_env, ckpt_path, config)
    new_logger = configure(str(log_path), ["stdout", "tensorboard"])

    # Create or load model
    if config.pretrained_model_file:
        # If the finetune model path exist, load the latest finetuned model,
        # otherwise load the pretrained model
        if ckpt_path.is_dir():
            latest_ckpt = get_latest_checkpoint_by_steps(ckpt_path)
            model = create_model(vec_env, config, latest_ckpt=latest_ckpt)
        else:
            model = create_model(vec_env, config, 
                                 load_ckpt=config.pretrained_model_file)
        model.set_logger(new_logger)
    else:
        latest_ckpt = get_latest_checkpoint_by_steps(ckpt_path)
        model = create_model(vec_env, config, latest_ckpt=latest_ckpt)
        model.set_logger(new_logger)
    
    # Train the model
    model.learn(total_timesteps=TRAINING_TIMESTEPS,
                callback=callbacks,
                reset_num_timesteps=False)

    # Save final model
    model.save(ckpt_path / "final_model")


if __name__ == '__main__':
    main()
