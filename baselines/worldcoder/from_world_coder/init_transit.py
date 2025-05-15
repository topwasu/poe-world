#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import copy

from .utils import extract_code_blocks, remove_duplicate_code, count_tokens_for_openai, get_avoid_words
from .transit_func_utils import experiences2text
from .evaluator import evaluate_transit_code
from ..env_info import DocString, TransitCodeExample

def init_transit(experiences, llm, verbose=True,):
    verbose_flag = verbose

    text_experiences = experiences2text(experiences)
    _experiences = copy.deepcopy(experiences)
    while count_tokens_for_openai(text_experiences) > 5120 and len(_experiences) > 1:
        _experiences = _experiences[:-1]
        text_experiences = experiences2text(_experiences)

    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            experiences=text_experiences,
            DocString=DocString,
            TransitCodeExample=TransitCodeExample,
        ),},
    ]
    if verbose_flag:
        print('-'*20 + 'Guessing initial code: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    llm_model_args = {'logit_bias': get_avoid_words(['class',])}
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=llm_model_args,)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Guessing Initial code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    code_blocks = extract_code_blocks(gen.content)
    while True:
        code = '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluate_transit_code(code, experiences,)
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
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'experiences': experiences,
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k,v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k,v in new_cb.usage.items() if k != '_lock'},
        'code': code,
    }
    return {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': code,
    }

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python. You will be provided experiences in the format of (state, action, next_state) tuples. You will also be provided with a short natural language description that briefly summarizes the difference between the state and the next state for each (state, next_state,) pair. You need to implement the python code to model the logic of the world, as seen in the provided experiences. Please follow the template to implement the code. The code needs to be directly runnable on the state and return the next state in python as provided in the experiences.
'''.strip()

FIRST_MESSAGE = '''
You need to implement python code to model the logic of the world as seen in the following experiences:

{experiences}

Please implement code to model the logic of the world as demonstrated by the experiences. Please implement the transition function following the template. The code needs to be directly runnable on the inputs of (state, action) and return the next state in python as provided in the experiences. You should only change the velocities of objects in the state. Please do not use any class such as ObjList to initialize the state. You can directly assign `new_state=state` to copy the state.

{DocString}

{TransitCodeExample}

Please implement code to model the logic of the world as demonstrated by the experiences. Please implement the code following the template. Feel free to implement the helper functions you need. You can also implement the logic for difference actions in different helper functions. However, you must implement the ` transition ` function as the main function to be called by the environment. The code needs to be directly runnable on the inputs as (state, action) and return the next state in python as provided in the experiences. Let's think step by step.
'''.strip()
