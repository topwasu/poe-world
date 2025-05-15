#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import copy

from .utils import extract_code_blocks, remove_duplicate_code, abbr_repr, count_tokens_for_openai, remove_noncompilable_code_blocks, get_avoid_words
from .transit_func_utils import experiences2text, experiences2text_with_wrong_outputs
from .evaluator import evaluate_transit_code
from ..env_info import DocString, TransitCodeExample

def refine_transit(
    code, crt_experiences, wrong_experiences, llm, verbose=True,
):
    verbose_flag = verbose

    crt_text_experiences = experiences2text(crt_experiences)
    _crt_experiences = copy.copy(crt_experiences)
    wrong_text_experiences = experiences2text_with_wrong_outputs(wrong_experiences, code)
    while count_tokens_for_openai(FIRST_MESSAGE.format(
        code = code,
        crt_experiences=crt_text_experiences,
        wrong_experiences=wrong_text_experiences,
        DocString=DocString,
        TransitCodeExample=TransitCodeExample,
    )) > 6500 and len(_crt_experiences) > 1:
        _crt_experiences = _crt_experiences[:-1]
        crt_text_experiences = experiences2text(_crt_experiences,)

    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            code = code,
            crt_experiences=crt_text_experiences,
            wrong_experiences=wrong_text_experiences,
            DocString=DocString,
            TransitCodeExample=TransitCodeExample,
        )},
    ]
    if verbose_flag:
        print('-'*20 + 'Refining  code: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    model_args = {'logit_bias': get_avoid_words()}
    gen = llm(chat_history, model_args=model_args,)
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=model_args,)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Refining code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    code_blocks = extract_code_blocks(gen.content)
    code_blocks = remove_noncompilable_code_blocks(code_blocks, prefix=code)
    while True:
        code = code + '\n' + '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluate_transit_code(code, crt_experiences + wrong_experiences,)
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
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'code': code,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k, v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k, v in new_cb.usage.items() if k != '_lock'},
        'code': code,
    }
    return {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': code,
    }

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python. You have tried it before and came up with one partially correct solution. However, it is not perfect. They can model the logic for some experiences but failed for others. You need to improve your code to model the logic of the world for all the experiences. The new code needs to be directly runnable on the (state, action) pair and return the next state in python as provided in the experiences.
'''


FIRST_MESSAGE = '''
{DocString}

{TransitCodeExample}

Here is the partially correct solution you came up with. It can model the logic for some experiences but failed for others. You need to improve your code to model the logic of the world for all the experiences. The new code needs to be directly runnable on the (state, action) pair and return the next state in python as provided in the experiences. You should only change the velocities of objects in the state. Please do not use any class such as ObjList to initialize the state. You can directly assign `new_state=state` to copy the state.

```

{code}

```

The given code cannot model the logic of the world for all the experiences. Here are some experiences that the code have successfully modeled.

{crt_experiences}

Here is an example of experiences that the code failed to model.

{wrong_experiences}

For this failed experience, do you know what is different between the true transitions from the environment and the predictions from the code? Do you know why the environment behaves in this way? Do you know why the code behaves differently from the environment? Which part of the code causes the problem? How to fix it? Please improve your code to model the logic of the world for all the experiences, accordingly. Please implement the code following the template. Feel free to implement any helper functions you need. You can also implement the logic for difference actions in different helper functions. However, you must implement the ` transition ` function as the main function to be called by the environment. The code needs to be directly runnable on the (state, action) tuple and return the new state in python as provided in the experiences. If the code is too long, try to refactor it to be shorter.
'''
