#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import copy
from gitinfo import get_git_info

from .utils import extract_code_blocks, remove_duplicate_code, abbr_repr, count_tokens_for_openai, remove_noncompilable_code_blocks, get_avoid_words
from .reward_utils import experiences2text, experiences2text_with_wrong_outputs
from .evaluator import RewardEvaluator

def refine_reward(
    lambda_code, crt_experiences, wrong_experiences, llm, verbose=True,
):
    mission = set([v['mission'] for v in wrong_experiences.values()])
    assert len(mission) == 1, 'All experiences should have the same mission.'
    mission = mission.pop()
    assert all([v['mission'] == mission for v in crt_experiences.values()]), 'All experiences should have the same mission.'
    mission = str(mission)
    assert isinstance(lambda_code, dict)
    assert mission in lambda_code, f'No code for mission {mission}.'
    code = lambda_code[mission]
    lambda_code = copy.deepcopy(lambda_code)

    verbose_flag = verbose
    wrong_text_experiences = experiences2text_with_wrong_outputs(wrong_experiences, code)
    if len(crt_experiences) == 0:
        crt_text_experiences = ''
    else:
        crt_text_experiences = experiences2text(crt_experiences)
        _crt_experiences = copy.deepcopy(crt_experiences)
        while count_tokens_for_openai(FIRST_MESSAGE.format(
            mission=mission,
            code = code,
            crt_experiences=crt_text_experiences,
            wrong_experiences=wrong_text_experiences,
        )) > 6500 and len(_crt_experiences) > 0:
            _crt_experiences.popitem()
            crt_text_experiences = experiences2text(_crt_experiences,)
        crt_text_experiences = 'The given code cannot model the logic of the world for all the experiences. Here are some experiences that the code has successfully modeled.\n\n' + crt_text_experiences

    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            mission=mission,
            code = code,
            crt_experiences=crt_text_experiences,
            wrong_experiences=wrong_text_experiences,
        ),},
    ]
    if verbose_flag:
        print('-'*20 + 'Refining  code: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    model_args = {'logit_bias': get_avoid_words()}
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=model_args,)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Refining code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    evaluator = RewardEvaluator()
    code_blocks = extract_code_blocks(gen.content)
    code_blocks = remove_noncompilable_code_blocks(code_blocks, prefix=code)
    while True:
        code = code + '\n' + '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluator(code, {**crt_experiences, **wrong_experiences},)
        if result['compilation_error'] is not None and len(code_blocks) > 1:
            if verbose_flag:
                print('Compilation Error:', result['compilation_error'])
            code_blocks = code_blocks[:-1]
        else:
            break
    if verbose_flag:
        print('\nResults:', abbr_repr(result))

    success_flag = result['success_flag']
    chat_history.append({'role': 'assistant', 'content': gen.content})
    lambda_code[mission] = code
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'lambda_code': lambda_code,
            'code': code,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'gitinfo': get_git_info(),
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k, v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k, v in new_cb.usage.items() if k != '_lock'},
        'mission': mission,
        '_code': code,
        'code': lambda_code,
    }
    return {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': lambda_code,
    }

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python. You have tried it before and came up with one partially correct solution. However, it is not perfect. They can model the logic for some experiences but failed for others. You need to improve your code to model the logic of the world for all the experiences. The new code needs to be directly runnable on the (state, action, next_state) tuple and return the (reward, done) tuple in python as provided in the experiences.
'''


FIRST_MESSAGE = '''
Here is the partially correct solution you came up with for mission "{mission}". It can model the logic for some experiences but failed for others. You need to improve your code to model the logic of the world for all the experiences. The new code need to be directly runnable on the (state, action, next_state) tuple and return the (reward, done) tuple in python as provided in the experiences.

```

{code}

```

{crt_experiences}

Here is an example of experiences that the code failed to model.

{wrong_experiences}

For this failed experience, do you know what is different between the true rewards and dones from the environment and the predictions from the code? Do you know why the environment behaves in this way? Do you know why the code behaves differently from the environment? Which part of the code causes the problem? How to fix it? Please improve your code to model the logic of the world for all the experiences, accordingly. Please implement the code following the template. You must implement the ` reward_func ` function as the main function to be called by the environment. The code needs to be directly runnable on the (state, action, next_state) tuple and return (reward, done) in python as provided in the experiences. If the code is too long, try to refactor it to be shorter.
'''
