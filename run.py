"""
Main entry point for running the active inference world model learning system.

Supports:
- World model synthesis from observations
- Model evaluation and visualization
- Interactive environment simulation
- Agent training and execution

Example command:
python run.py \
load_obj_model=saved_checkpoints_storage/reasoner_player_final_basic9_newest.pickle \
det_world_model=True obs_suffix=_basic9 run_world_model=False \
moe.batch_size=10000 moe.lr=1 moe.optim=lbfgs moe.n_steps=2 \
agent.initial_budget_iterations=4000 agent.slow_budget_increase=True
"""
from typing import List
import os
import random

import hydra
import dill as pickle
import logging
import numpy as np
from omegaconf import DictConfig

from baselines.worldcoder import WorldCoder
from learners.world_model_learner import PoEWorldLearner, WorldModelLearner
from agents.agent import Agent
from data.atari import load_atari_observations
from classes.envs import *
from classes.envs.object_tracker import *
from classes.envs.renderer import get_human_renderer
from classes.helper import set_global_constants, StateTransitionTriplet
from openai_hf_interface import choose_provider
from eval import evaluate_world_model

log = logging.getLogger('main')
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """
    Main execution function that:
    1. Sets up logging and environment
    2. Either loads or synthesizes a world model
    3. Runs interactive simulation or trains an agent
    4. Evaluates model performance
    
    Args:
        config: Hydra configuration object containing all parameters
    """
    # # Setup logging
    # log = configure_logging(config.debug_mode)

    # Configure OpenAI/HuggingFace API
    choose_provider(config.provider)

    # Set random seeds
    set_seed(config.random_seed)

    # Initialize game-specific constants
    set_global_constants(config.task)
    
    if config.database_path is None:
        config.database_path = f'completions_atari_{config.task.lower()}{"" if config.seed == 0 else f"_s{config.seed}"}.db'

    # --- Synthesize world model ---
    # Load observations -- use the same observation for both non-prime and prime versions
    observations, actions, game_states = load_atari_observations(
        config.task.replace('Alt', '') + config.obs_suffix)

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
                                                    output_game_state=game_states[i + 1]))

    learner = get_world_model_learner(config)
    world_model = learner.synthesize_world_model(transitions)
    world_model.clear_cache()

    # Either run interactive simulation or train agent
    if config.post_synthesis_mode == 'run':
        object_tracker = ObjectTracker()
        renderer = get_human_renderer(config)
        imagine_env = ImaginedAtariEnv(config, world_model, object_tracker,
                                       renderer)
        env_player = EnvPlayer(imagine_env)
        env_player.run()
    elif config.post_synthesis_mode == 'agent':
        agent = Agent(config, learner)
        agent.plan_and_execute(get_goal_obj_type_by_game(config))
    elif config.post_synthesis_mode == 'evaluate':
        evaluate_world_model(config, world_model)
    elif config.post_synthesis_mode == 'nothing':
        log.info("Not doing anything post-synthesis")
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
