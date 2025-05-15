#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import time
import copy
from gitinfo import get_git_info

from .utils import extract_code_blocks, remove_duplicate_code, count_tokens_for_openai, get_avoid_words
from .reward_utils import get_func_template, experiences2text
from .evaluator import RewardEvaluator

def init_reward(experiences, env_metadata, llm, verbose=True,):
    assert isinstance(experiences, dict)
    mission = set([v['mission'] for v in experiences.values()])
    assert len(mission) == 1, f'Experiences should have the same mission, but got {mission}'
    mission = str(mission.pop())
    output_reward_code = dict()
    verbose_flag = verbose

    text_experiences = experiences2text(experiences)
    _experiences = copy.deepcopy(experiences)
    while count_tokens_for_openai(text_experiences) > 5120 and len(_experiences) > 1:
        _experiences.pop(sorted(_experiences.keys(), key=str)[0])
        text_experiences = experiences2text(_experiences)

    func_template = get_func_template(env_metadata)
    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            mission=mission,
            experiences=text_experiences,
            code_template=func_template,
        )},
    ]
    if verbose_flag:
        print('-'*20 + 'Guessing initial LambdaReward code: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    model_args = {'logit_bias': get_avoid_words(['class',])}
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=model_args)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Guessing initial LambdaReward code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    evaluator = RewardEvaluator()
    code_blocks = extract_code_blocks(gen.content)
    while True:
        code = func_template + '\n' + '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluator(code, experiences,)
        if result['compilation_error'] is not None and len(code_blocks) > 1:
            if verbose_flag:
                print('Compilation Error:', result['compilation_error'])
            code_blocks = code_blocks[:-1]
        else:
            break
    if verbose_flag:
        print('\nResults:', {
            k: v for k, v in result.items()
            if len(str(v)) < 100
        })

    success_flag = result['success_flag']
    chat_history.append({'role': 'assistant', 'content': gen.content})
    output_reward_code[str(mission)] = code
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'env_metadata': env_metadata,
            'experiences': experiences,
            'gitinfo': get_git_info(),
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k,v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k,v in new_cb.usage.items() if k != '_lock'},
        'code': output_reward_code,
    }
    output = {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': output_reward_code,
    }
    return output

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python. You will be provided experiences in the format of (state, action, next_state, reward, done) tuples. You will also be provided with a short natural language description that briefly summarizes the difference between the state and the next state for each (state, next_state) pair. You need to implement the python code to model the logic of the world, as seen in the provided experiences. Please follow the template to implement the code. The code needs to be directly runnable on the (state, action, next_state) tuple and return the (reward, done) tuple in python as provided in the experiences.
'''

FIRST_MESSAGE = '''
You need to implement python code to model the logic of the world as seen in the following experiences for mission "{mission}":

{experiences}

Please implement code to model the logic of the world as demonstrated by the experiences. Here is the template for the reward function. Please implement the reward function following the template. The code needs to be directly runnable on the inputs of (state, action, next_state) and return (reward, done) in python as provided in the experiences.

```

{code_template}

```

Please implement code to model the logic of the world as demonstrated by the experiences. Please implement the code following the template. You must implement the ` reward_func ` function as the main function to be called by the environment. The code needs to be directly runnable on the inputs as (state, action, next_state) and return (reward, done) in python as provided in the experiences. Let's think step by step.
'''
