
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

from data.atari import load_atari_observations
from classes.envs import *
from classes.envs.object_tracker import *
from classes.envs.renderer import get_human_renderer
from classes.helper import set_global_constants
from openai_hf_interface import choose_provider

log = logging.getLogger('main')
log.setLevel(logging.INFO)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # Configure OpenAI/HuggingFace API
    choose_provider(config.provider)

    # Set random seeds
    set_seed(config.seed)

    # Initialize game-specific constants
    set_global_constants(config.task)

    # Load observations
    observations, actions, game_states = load_atari_observations(
        config.task + config.obs_suffix)

    # Optional: Use subset of observations
    if config.obs_index != -1:
        observations = observations[config.obs_index:config.obs_index +
                                    config.obs_index_length + 1]
        actions = actions[config.obs_index:config.obs_index +
                            config.obs_index_length]
        game_states = game_states[config.obs_index:config.obs_index +
                                    config.obs_index_length + 1]

    renderer = get_human_renderer(config)
    for idx, obs in enumerate(observations):
        renderer.render(obs)
        balls = obs.get_objs_by_obj_type("ball")
        if balls:
            log.info(f'idx {idx}')
        if idx == 580:
            breakpoint()


if __name__ == '__main__':
    main()
