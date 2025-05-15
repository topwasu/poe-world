#!/usr/bin/env python
# coding=utf-8

import copy
from types import SimpleNamespace

from ...utils.eval_code import eval_code
from .evaluator import RewardEvaluator

REWARD_FUNC_TEMPLATE = '''
def reward_func(state, action, next_state):
    """
    Args:
        state: the state of the environment
        action: the action to be executed
        next_state: the next state of the environment
    Returns:
        reward: the reward of the action
        done: whether the episode is done
    """
    raise NotImplementedError
'''.strip()

def get_func_template(env_metadata):
    func_template = copy.deepcopy(REWARD_FUNC_TEMPLATE)
    api = env_metadata['api']
    func_template = api + '\n' + func_template
    return func_template

def experiences2text(experiences):
    keys = list(experiences.keys())
    old_state_list = [experiences[k]['state'] for k in keys]
    action_list = [experiences[k]['action'] for k in keys]
    new_state_list = [experiences[k]['state_next'] for k in keys]
    reward_list = [experiences[k]['reward'] for k in keys]
    done_list = [experiences[k]['done'] for k in keys]
    difference_list = [experiences[k]['state_next'] - experiences[k]['state'] for k in keys]
    text_experiences = [
        f'The action "{action}" transforms the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n, the returned reward is ` {reward} ` and the returned done is ` {done} `'.strip()
        for old_state, action, new_state, reward, done, diff in zip(old_state_list, action_list, new_state_list, reward_list, done_list, difference_list)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences

def experiences2text_with_wrong_outputs(experiences, code):
    result = RewardEvaluator()(code, experiences)
    result_list = result['result_list']
    # assert all([not res['success_flag'] for res in result_list]), (f'Expect all results to be wrong, but got {result_list}', code)
    pred_reward_list = [res['pred_reward'] if res['pred_reward'] is not None else None for res in result_list]
    pred_done_list = [res['pred_done'] if res['pred_done'] is not None else None for res in result_list]

    keys = list(experiences.keys())
    old_state_list = [experiences[k]['state'] for k in keys]
    action_list = [experiences[k]['action'] for k in keys]
    new_state_list = [experiences[k]['state_next'] for k in keys]
    reward_list = [experiences[k]['reward'] for k in keys]
    done_list = [experiences[k]['done'] for k in keys]
    difference_list = [experiences[k]['state_next'] - experiences[k]['state'] for k in keys]
    compilation_error_list = [res['compilation_error'] for res in result_list]
    assert len(old_state_list) == len(new_state_list) == len(difference_list) == len(action_list) == len(reward_list) == len(done_list) == len(pred_reward_list) == len(pred_done_list), f'Expect all lists to have the same length, but got {len(old_state_list)}, {len(new_state_list)}, {len(difference_list)}, {len(action_list)}, {len(reward_list)}, {len(done_list)}, {len(pred_reward_list)}, {len(pred_done_list)}'

    text_experiences = [
        _exp2text_wrong(old_state, action, new_state, reward, done, diff, pred_reward, pred_done, compilation_error)
        for old_state, action, new_state, reward, done, diff, pred_reward, pred_done, compilation_error in zip(old_state_list, action_list, new_state_list, reward_list, done_list, difference_list, pred_reward_list, pred_done_list, compilation_error_list)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences
def _exp2text_wrong(old_state, action, new_state, reward, done, diff, pred_reward, pred_done, compilation_error):
    if compilation_error is not None:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\nHowever, the implementation fails to compile with error\n```\n{compilation_error[:compilation_error.rfind("Printed outputs:")].strip()}\n```\n'.strip()
    elif pred_done != done:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n, the returned reward should be ` {reward} ` and the returned done should be ` {done} `.\nHowever, the implementation is wrong because it returns the predicted done as ` {pred_done} ` instead of the correct done as ` {done} `.'.strip()
    elif pred_reward != reward:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n, the returned reward should be ` {reward} ` and the returned done should be ` {done} `.\nHowever, the implementation is wrong because it returns the predicted reward as ` {pred_reward} ` instead of the correct reward as ` {reward} `.'.strip()
    else:
        # To compensate for the different evaluation in just reward v.s. reward +
        # transit. Should be corrected but since it is only making performance
        # worse, so we will ignore it for now as it is not conflicting with our
        # claim (if not making our claim stronger).
        print(f'Evaluating the implementation with the correct reward and done, but got correct implementation')
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n, the returned reward should be ` {reward} ` and the returned done should be ` {done} `.'.strip()
        raise ValueError(f'Expect the implementation to be wrong, but got correct implementation')


