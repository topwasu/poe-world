
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
from openai_hf_interface import choose_provider, create_llm
from eval import evaluate_world_model

log = logging.getLogger('main')
log.setLevel(logging.INFO)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    

instruction_prompt_mr = """
You are playing a video game. Your goal is to collect the key. 
The actions you can take are: NOOP, UP, DOWN, RIGHT, LEFT, FIRE, LEFTFIRE, RIGHTFIRE.
Additionally, you can think about the game state, by saying "THINK: <your thoughts>" and take action later.

Here are examples of how to play the game:

OBSERVATION: 

List of objects:
player object (id = 0) with x=76, y=73
key object (id = 1) with x=13, y=99
skull object (id = 2) with x=90, y=166
barrier object (id = 3) with x=20, y=54
barrier object (id = 4) with x=136, y=54
rope object (id = 5) with x=112, y=96
platform object (id = 6) with x=0, y=93
platform object (id = 7) with x=104, y=93
platform object (id = 8) with x=68, y=93
ladder object (id = 9) with x=72, y=93
platform object (id = 10) with x=76, y=136
conveyer_belt object (id = 11) with x=60, y=136
conveyer_belt object (id = 12) with x=85, y=136
ladder object (id = 13) with x=16, y=136
ladder object (id = 14) with x=128, y=136
platform object (id = 15) with x=8, y=136
platform object (id = 16) with x=124, y=136
platform object (id = 17) with x=16, y=180
wall object (id = 18) with x=0, y=96
wall object (id = 19) with x=0, y=136
wall object (id = 20) with x=152, y=96
wall object (id = 21) with x=140, y=136
life object (id = 22) with x=56, y=15
life object (id = 23) with x=64, y=15
life object (id = 24) with x=72, y=15
life object (id = 25) with x=80, y=15
life object (id = 26) with x=88, y=15
score object (id = 27) with x=97, y=6

List of object interactions:
player (id = 0) touches platform (id = 8) with touch_side=bottom and touch_percent=1.0
player (id = 0) touches ladder (id = 9) with touch_side=bottom and touch_percent=1.0
platform (id = 8) touches player (id = 0) with touch_side=top and touch_percent=0.6
platform (id = 8) touches ladder (id = 9) with touch_side=left and touch_percent=1.0
ladder (id = 9) touches player (id = 0) with touch_side=top and touch_percent=0.6
ladder (id = 9) touches platform (id = 8) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 15) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 13) touches wall (id = 19) with touch_side=left and touch_percent=1.0
ladder (id = 14) touches platform (id = 16) with touch_side=top and touch_percent=1.0
ladder (id = 14) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 14) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 15) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 15) touches wall (id = 18) with touch_side=overlap and touch_percent=0
platform (id = 15) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 16) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
platform (id = 16) touches wall (id = 20) with touch_side=overlap and touch_percent=0
platform (id = 16) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 17) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 17) touches ladder (id = 14) with touch_side=right and touch_percent=1.0
platform (id = 17) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 17) touches wall (id = 21) with touch_side=right and touch_percent=1.0
wall (id = 18) touches platform (id = 15) with touch_side=overlap and touch_percent=0
wall (id = 19) touches ladder (id = 13) with touch_side=right and touch_percent=1.0
wall (id = 19) touches platform (id = 15) with touch_side=top and touch_percent=0.5
wall (id = 19) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1
wall (id = 20) touches platform (id = 16) with touch_side=overlap and touch_percent=0
wall (id = 21) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
wall (id = 21) touches platform (id = 16) with touch_side=top and touch_percent=0.5
wall (id = 21) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1

ACTION: THINK: The player is at (76, 73) and the key is at (13, 99).
Currently, the player is on top of a platform (id = 8) and a ladder (id = 9). 
To reach the key, the player can climb down the ladder to get closer to the key.

OBSERVATION: OK.

ACTION: DOWN

OBSERVATION: 

List of objects:
player object (id = 0) with x=76, y=84
key object (id = 1) with x=13, y=99
skull object (id = 2) with x=89, y=166
barrier object (id = 3) with x=20, y=54
barrier object (id = 4) with x=136, y=54
rope object (id = 5) with x=112, y=96
platform object (id = 6) with x=0, y=93
platform object (id = 7) with x=104, y=93
platform object (id = 8) with x=68, y=93
ladder object (id = 9) with x=72, y=93
platform object (id = 10) with x=76, y=136
conveyer_belt object (id = 11) with x=60, y=136
conveyer_belt object (id = 12) with x=85, y=136
ladder object (id = 13) with x=16, y=136
ladder object (id = 14) with x=128, y=136
platform object (id = 15) with x=8, y=136
platform object (id = 16) with x=124, y=136
platform object (id = 17) with x=16, y=180
wall object (id = 18) with x=0, y=96
wall object (id = 19) with x=0, y=136
wall object (id = 20) with x=152, y=96
wall object (id = 21) with x=140, y=136
life object (id = 22) with x=56, y=15
life object (id = 23) with x=64, y=15
life object (id = 24) with x=72, y=15
life object (id = 25) with x=80, y=15
life object (id = 26) with x=88, y=15
score object (id = 27) with x=97, y=6

List of object interactions:
player (id = 0) touches platform (id = 8) with touch_side=left and touch_percent=0.5
player (id = 0) touches ladder (id = 9) with touch_side=bottom and touch_percent=1.0
platform (id = 8) touches player (id = 0) with touch_side=left and touch_percent=1.0
platform (id = 8) touches ladder (id = 9) with touch_side=left and touch_percent=1.0
ladder (id = 9) touches player (id = 0) with touch_side=top and touch_percent=0.6
ladder (id = 9) touches platform (id = 8) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 15) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 13) touches wall (id = 19) with touch_side=left and touch_percent=1.0
ladder (id = 14) touches platform (id = 16) with touch_side=top and touch_percent=1.0
ladder (id = 14) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 14) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 15) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 15) touches wall (id = 18) with touch_side=overlap and touch_percent=0
platform (id = 15) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 16) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
platform (id = 16) touches wall (id = 20) with touch_side=overlap and touch_percent=0
platform (id = 16) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 17) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 17) touches ladder (id = 14) with touch_side=right and touch_percent=1.0
platform (id = 17) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 17) touches wall (id = 21) with touch_side=right and touch_percent=1.0
wall (id = 18) touches platform (id = 15) with touch_side=overlap and touch_percent=0
wall (id = 19) touches ladder (id = 13) with touch_side=right and touch_percent=1.0
wall (id = 19) touches platform (id = 15) with touch_side=top and touch_percent=0.5
wall (id = 19) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1
wall (id = 20) touches platform (id = 16) with touch_side=overlap and touch_percent=0
wall (id = 21) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
wall (id = 21) touches platform (id = 16) with touch_side=top and touch_percent=0.5
wall (id = 21) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1

ACTION: THINK: The player is at (76, 84) and the key is at (13, 99).
Currently, the player is on right of a platform (id = 8) and on top of a ladder (id = 9). 
To reach the key, the player can climb down the ladder to get closer to the key.

OBSERVATION: OK.

ACTION: DOWN

OBSERVATION: 

List of objects:
player object (id = 0) with x=76, y=87
key object (id = 1) with x=13, y=99
skull object (id = 2) with x=88, y=166
barrier object (id = 3) with x=20, y=54
barrier object (id = 4) with x=136, y=54
rope object (id = 5) with x=112, y=96
platform object (id = 6) with x=0, y=93
platform object (id = 7) with x=104, y=93
platform object (id = 8) with x=68, y=93
ladder object (id = 9) with x=72, y=93
platform object (id = 10) with x=76, y=136
conveyer_belt object (id = 11) with x=60, y=136
conveyer_belt object (id = 12) with x=85, y=136
ladder object (id = 13) with x=16, y=136
ladder object (id = 14) with x=128, y=136
platform object (id = 15) with x=8, y=136
platform object (id = 16) with x=124, y=136
platform object (id = 17) with x=16, y=180
wall object (id = 18) with x=0, y=96
wall object (id = 19) with x=0, y=136
wall object (id = 20) with x=152, y=96
wall object (id = 21) with x=140, y=136
life object (id = 22) with x=56, y=15
life object (id = 23) with x=64, y=15
life object (id = 24) with x=72, y=15
life object (id = 25) with x=80, y=15
life object (id = 26) with x=88, y=15
score object (id = 27) with x=97, y=6

List of object interactions:
player (id = 0) touches platform (id = 8) with touch_side=top and touch_percent=1.0
player (id = 0) touches ladder (id = 9) with touch_side=top and touch_percent=1.0
platform (id = 8) touches player (id = 0) with touch_side=left and touch_percent=1.0
platform (id = 8) touches ladder (id = 9) with touch_side=left and touch_percent=1.0
ladder (id = 9) touches player (id = 0) with touch_side=top and touch_percent=0.6
ladder (id = 9) touches platform (id = 8) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 15) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 13) touches wall (id = 19) with touch_side=left and touch_percent=1.0
ladder (id = 14) touches platform (id = 16) with touch_side=top and touch_percent=1.0
ladder (id = 14) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 14) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 15) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 15) touches wall (id = 18) with touch_side=overlap and touch_percent=0
platform (id = 15) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 16) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
platform (id = 16) touches wall (id = 20) with touch_side=overlap and touch_percent=0
platform (id = 16) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 17) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 17) touches ladder (id = 14) with touch_side=right and touch_percent=1.0
platform (id = 17) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 17) touches wall (id = 21) with touch_side=right and touch_percent=1.0
wall (id = 18) touches platform (id = 15) with touch_side=overlap and touch_percent=0
wall (id = 19) touches ladder (id = 13) with touch_side=right and touch_percent=1.0
wall (id = 19) touches platform (id = 15) with touch_side=top and touch_percent=0.5
wall (id = 19) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1
wall (id = 20) touches platform (id = 16) with touch_side=overlap and touch_percent=0
wall (id = 21) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
wall (id = 21) touches platform (id = 16) with touch_side=top and touch_percent=0.5
wall (id = 21) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1

ACTION: THINK: The player is at (76, 87) and the key is at (13, 99).
Currently, the player is below a platform (id = 8) and below a ladder (id = 9). 
Maybe I can try to move horizontally towards the key.

OBSERVATION: OK.

ACTION: LEFT

OBSERVATION: 

List of objects:
player object (id = 0) with x=76, y=87
key object (id = 1) with x=13, y=99
skull object (id = 2) with x=87, y=166
barrier object (id = 3) with x=20, y=54
barrier object (id = 4) with x=136, y=54
rope object (id = 5) with x=112, y=96
platform object (id = 6) with x=0, y=93
platform object (id = 7) with x=104, y=93
platform object (id = 8) with x=68, y=93
ladder object (id = 9) with x=72, y=93
platform object (id = 10) with x=76, y=136
conveyer_belt object (id = 11) with x=60, y=136
conveyer_belt object (id = 12) with x=85, y=136
ladder object (id = 13) with x=16, y=136
ladder object (id = 14) with x=128, y=136
platform object (id = 15) with x=8, y=136
platform object (id = 16) with x=124, y=136
platform object (id = 17) with x=16, y=180
wall object (id = 18) with x=0, y=96
wall object (id = 19) with x=0, y=136
wall object (id = 20) with x=152, y=96
wall object (id = 21) with x=140, y=136
life object (id = 22) with x=56, y=15
life object (id = 23) with x=64, y=15
life object (id = 24) with x=72, y=15
life object (id = 25) with x=80, y=15
life object (id = 26) with x=88, y=15
score object (id = 27) with x=97, y=6

List of object interactions:
player (id = 0) touches platform (id = 8) with touch_side=top and touch_percent=1.0
player (id = 0) touches ladder (id = 9) with touch_side=top and touch_percent=1.0
platform (id = 8) touches player (id = 0) with touch_side=left and touch_percent=1.0
platform (id = 8) touches ladder (id = 9) with touch_side=left and touch_percent=1.0
ladder (id = 9) touches player (id = 0) with touch_side=top and touch_percent=0.6
ladder (id = 9) touches platform (id = 8) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 15) with touch_side=top and touch_percent=1.0
ladder (id = 13) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 13) touches wall (id = 19) with touch_side=left and touch_percent=1.0
ladder (id = 14) touches platform (id = 16) with touch_side=top and touch_percent=1.0
ladder (id = 14) touches platform (id = 17) with touch_side=bottom and touch_percent=1.0
ladder (id = 14) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 15) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 15) touches wall (id = 18) with touch_side=overlap and touch_percent=0
platform (id = 15) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 16) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
platform (id = 16) touches wall (id = 20) with touch_side=overlap and touch_percent=0
platform (id = 16) touches wall (id = 21) with touch_side=right and touch_percent=1.0
platform (id = 17) touches ladder (id = 13) with touch_side=left and touch_percent=1.0
platform (id = 17) touches ladder (id = 14) with touch_side=right and touch_percent=1.0
platform (id = 17) touches wall (id = 19) with touch_side=left and touch_percent=1.0
platform (id = 17) touches wall (id = 21) with touch_side=right and touch_percent=1.0
wall (id = 18) touches platform (id = 15) with touch_side=overlap and touch_percent=0
wall (id = 19) touches ladder (id = 13) with touch_side=right and touch_percent=1.0
wall (id = 19) touches platform (id = 15) with touch_side=top and touch_percent=0.5
wall (id = 19) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1
wall (id = 20) touches platform (id = 16) with touch_side=overlap and touch_percent=0
wall (id = 21) touches ladder (id = 14) with touch_side=left and touch_percent=1.0
wall (id = 21) touches platform (id = 16) with touch_side=top and touch_percent=0.5
wall (id = 21) touches platform (id = 17) with touch_side=bottom and touch_percent=0.1

ACTION: THINK: The player is at (76, 87) and the key is at (13, 99).
It seems like I can't move horizontally towards the key while being below a platform and a ladder.
Let's keep moving down the ladder.

OBSERVATION: OK.

ACTION: DOWN

Now, Here is the real history of observations and actions, either take an ACTION: <action> or ACTION: THINK: <thoughts> about the game state. DO NOT generate OBSERVATION: in your response:


"""


instruction_prompt_pong = """
You are playing a video game. Your goal is to use hit the ball with the paddle (player) to the enemy.
The actions you can take are: NOOP, RIGHT, LEFT, FIRE, LEFTFIRE, RIGHTFIRE.
Additionally, you can think about the game state, by saying "THINK: <your thoughts>" and take action later.
However, you should not generate ACTION: THINK: more than three times in a row.

Here are examples of how to play the game:

OBSERVATION: 

List of objects:
player object (id = 0) with x=140, y=96
ball object (id = 7) with x=51, y=110
enemy object (id = 8) with x=16, y=117
playerscore object (id = 1) with x=116, y=1
enemyscore object (id = 2) with x=36, y=1
wall object (id = 3) with x=6, y=29
wall object (id = 4) with x=6, y=194
zone object (id = 5) with x=0, y=29
zone object (id = 6) with x=155, y=29

List of object interactions:
No touching objects

ACTION: THINK: The player is at (140, 96) and the ball is at (51, 110).
The ball is below the player, let's move towards the ball.

OBSERVATION: OK.

ACTION: LEFT

OBSERVATION: 

List of objects:
player object (id = 0) with x=140, y=98
ball object (id = 7) with x=54, y=104
enemy object (id = 8) with x=16, y=111
playerscore object (id = 1) with x=116, y=1
enemyscore object (id = 2) with x=36, y=1
wall object (id = 3) with x=6, y=29
wall object (id = 4) with x=6, y=194
zone object (id = 5) with x=0, y=29
zone object (id = 6) with x=155, y=29

List of object interactions:
No touching objects

ACTION: THINK: The player is at (140, 98) and the ball is at (54, 104).
The ball is below the player, let's move towards the ball.

OBSERVATION: OK.

ACTION: LEFT

OBSERVATION: 

List of objects:
player object (id = 0) with x=140, y=115
ball object (id = 7) with x=57, y=98
enemy object (id = 8) with x=16, y=107
playerscore object (id = 1) with x=116, y=1
enemyscore object (id = 2) with x=36, y=1
wall object (id = 3) with x=6, y=29
wall object (id = 4) with x=6, y=194
zone object (id = 5) with x=0, y=29
zone object (id = 6) with x=155, y=29

List of object interactions:
No touching objects

ACTION: THINK: The player is at (140, 115) and the ball is at (57, 98).
The ball is above the player, let's move towards the ball.

OBSERVATION: OK.

ACTION: RIGHT

OBSERVATION:

List of objects:
player object (id = 0) with x=140, y=120
ball object (id = 7) with x=60, y=92
enemy object (id = 8) with x=16, y=101
playerscore object (id = 1) with x=116, y=1
enemyscore object (id = 2) with x=36, y=1
wall object (id = 3) with x=6, y=29
wall object (id = 4) with x=6, y=194
zone object (id = 5) with x=0, y=29
zone object (id = 6) with x=155, y=29

List of object interactions:
No touching objects

ACTION: THINK: The player is at (140, 120) and the ball is at (60, 92).
The ball is above the player, let's move towards the ball.

OBSERVATION: OK.

ACTION: RIGHT

Now, Here is the real history of observations and actions, either take an ACTION: <action> or ACTION: THINK: <thoughts> about the game state. 
DO NOT generate OBSERVATION: in your response
And DO NOT generate ACTION: THINK: more than three times in a row.


"""
    
    
def run_react(config, atari_env):
    # Create llm
    cache_mode = 'disk_to_memory' if config.use_memory else 'disk'
    llm = create_llm('gpt-4o-2024-08-06' if config.provider ==
                        'openai' else 'openai/gpt-4o-2024-08-06')
    llm.setup_cache(cache_mode, database_path=config.database_path)
    llm.set_default_kwargs({'timeout': 60})
    
    # Prompt
    instruction_prompt = instruction_prompt_mr if config.task.startswith('MontezumaRevenge') else instruction_prompt_pong
    
    obj_list, game_state = atari_env.reset()
    # for _ in range(50):
    #     obj_list, game_state = atari_env.step('NOOP')
        
    # obj_list, game_state = atari_env.step('LEFT')
    # obj_list, game_state = atari_env.step('LEFT')
    # obj_list, game_state = atari_env.step('RIGHT')
    # log.info('OBSERVATION:\n\n{}'.format(obj_list.get_str_w_ints_w_touching(w_id=True)))
    # breakpoint()
    
    history_txt_lst = [f'OBSERVATION:\n\n{obj_list.get_str_w_ints_w_touching(w_id=True)}']
    all_actions = []
    ct = 0
    while True:
        ct += 1
        if len(history_txt_lst) > 16:
            history_txt_lst = history_txt_lst[-16:]
            
        outputs = llm.prompt([instruction_prompt + "\n\n".join(history_txt_lst) + '\n\n'], temperature=0, seed=config.seed)
        log.info(f'Outputs: {outputs}')
        action = outputs[0].strip().split('ACTION:')[1].strip().split('\n')[0]
        if action.startswith('THINK:'):
            observation = 'OBSERVATION: OK.'
        else:
            obj_list, game_state = atari_env.step(action)
            observation = f'OBSERVATION:\n\n{obj_list.get_str_w_ints_w_touching(w_id=True)}'
        
        history_txt_lst.append(f'ACTION: {action}')
        history_txt_lst.append(observation)
        
        log.info(f'TOOK ACTION: {action}')
        
        # if ct > 620:
        #     log.info(f'TOOK TOO MANY ACTIONS')
        #     breakpoint()
        #     break
        
        if action.startswith('THINK:'):
            continue
        
        all_actions.append(action)
        if game_state == GameState.GAMEOVER:
            log.info(f'GAME OVER')
            break
        
        if llm.get_info()['actual_cost'] > 30:
            log.info(f'EXCEEDED COST LIMIT')
            break
            
        
        log.info(f'Iterations: {ct}')
        log.info(f'Actions taken: {len(all_actions)}')
        log.info(f'Last 10 actions: {all_actions[-10:]}')
        # txt = "\n\n".join(history_txt_lst)
        # log.info(f'History: {txt}')
        
        log.info(f'LLM cost: {llm.get_info()}')
        
    
    log.info(f"Total actions: {len(all_actions)}")
    log.info(f"Actions: {all_actions}")
    

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
    set_seed(config.seed)

    # Initialize game-specific constants
    set_global_constants(config.task)
    
    if config.database_path is None:
        config.database_path = f'completions_atari_{config.task.lower()}_react{"" if config.seed == 0 else f"_s{config.seed}"}.db'

    renderer = get_human_renderer(config)
    atari_env = create_atari_env(config, config.task, renderer=renderer, skip_gameover_if_possible=False)
    run_react(config, atari_env)


if __name__ == '__main__':
    main()
