#!/usr/bin/env python
# coding=utf-8

import copy
from classes.helper import StateTransitionTriplet, are_two_obj_lists_equal, ObjList

from .eval_code import eval_code

def get_transit_func(code):
    compilation_error, func_name, transit_func = None, None, None
    exec_globals = {}
    exec_globals = eval_code(code, exec_globals=exec_globals, return_exec_globals=True)
    if not isinstance(exec_globals, dict):
        compilation_error = exec_globals
        return compilation_error, func_name, transit_func, exec_globals

    if 'transition' in exec_globals and callable(exec_globals['transition']):
        func_name = {'transition': exec_globals['transition']}
    else:
        func_name = {k:v for k, v in exec_globals.items() if not k.startswith('_') and callable(v)}
        if len(func_name) == 0:
            compilation_error = 'No transition function found'
            return compilation_error, func_name, transit_func, exec_globals
        elif len(func_name) > 1:
            tmp_func_name = {k:v for k, v in func_name.items() if 'transit' in k}
            if len(tmp_func_name) >= 1:
                func_name = tmp_func_name
        if len(func_name) > 1:
            print(f'Warning: Expect only one transition function, but got {len(func_name)}')
            lastest_k = list(func_name.keys())[-1]
            func_name = {lastest_k: func_name[lastest_k]}
    func_name = list(func_name.keys())[0]
    transit_func = exec_globals[func_name]
    assert callable(transit_func), f'Expect {func_name} to be callable, but got {transit_func}'
    return compilation_error, func_name, transit_func, exec_globals
def evaluate_transit_code(code, experiences):
    compilation_error, func_name, transit_func, exec_globals = get_transit_func(code)
    if compilation_error is not None:
        results = {
            'success_flag': False,
            'success_ratio': 0,
            'compilation_error': compilation_error,
            'crt_experiences': dict(),
            'wrong_experiences': experiences,
            'experiences': experiences,
            'result_list': [{
                'success_flag': False,
                'pred_new_state': None,
                'pred_state_success_flag': False,
                'experience': exp,
                'compilation_error': compilation_error,
            } for exp in experiences],
            'func_name': None,
        }
        return results

    result_list = [eval_transit_per_experience(transit_func, x, exec_globals,) for x in experiences]
    success_flag = all([result['success_flag'] for result in result_list])
    success_ratio = sum([result['success_flag'] for result in result_list]) / len(result_list)
    pred_state_success_flag = all([result['pred_state_success_flag'] for result in result_list])
    pred_state_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
    crt_experiences = [x for x, result in zip(experiences, result_list) if result['success_flag']]
    wrong_experiences = [x for x, result in zip(experiences, result_list) if not result['success_flag']]
    results = {
        'success_flag': success_flag,
        'success_ratio': success_ratio,
        'pred_state_success_flag': pred_state_success_flag,
        'pred_state_success_ratio': pred_state_success_ratio,
        'compilation_error': None,
        'crt_experiences': crt_experiences,
        'wrong_experiences': wrong_experiences,
        'experiences': experiences,
        'result_list': result_list,
        'func_name': func_name,
    }
    return results
def eval_transit_per_experience(transit_func, experience, exec_globals):
    # Copy the old state
    code_to_run = "old_state = experience.input_state.deepcopy()"
    _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
    if isinstance(_exec_globals, str):
        return {
            'success_flag': False,
            'pred_state_success_flag': False,
            'pred_new_state': None,
            'experience': experience,
            'compilation_error': _exec_globals,
        }
    old_state = _exec_globals['old_state']

    # Copy the action
    code_to_run = "action = experience.event"
    _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
    if isinstance(_exec_globals, str):
        return {
            'success_flag': False,
            'pred_state_success_flag': False,
            'pred_new_state': None,
            'experience': experience,
            'compilation_error': _exec_globals,
        }
    action = _exec_globals['action']
    new_state = experience.output_state

    # Copy the transit function
    exec_globals.update({
        'copied_old_state': old_state,
        'copied_action': action,
        'transit_func': transit_func,
    })

    # Old state pre_step
    code_to_run = "copied_old_state.pre_step()"
    _exec_globals = ''
    while isinstance(_exec_globals, str):
        _exec_globals = eval_code(code_to_run, exec_globals=exec_globals, return_exec_globals=True, timeout=0.5)    
    assert not isinstance(_exec_globals, str), f'Failed to run the pre_step of the old state: {_exec_globals}'

    # Run the transit function
    code_to_exec = 'pred_new_state = transit_func(copied_old_state, copied_action)'
    exec_globals = eval_code(code_to_exec, exec_globals=exec_globals, return_exec_globals=True)
    if isinstance(exec_globals, str):
        return {
            'success_flag': False,
            'pred_state_success_flag': False,
            'pred_new_state': None,
            'experience': experience,
            'compilation_error': exec_globals,
        }

    # Check the new state
    pred_new_state = exec_globals['pred_new_state']
    valid_state = isinstance(pred_new_state, new_state.__class__)
    if not valid_state:
        return {
            'success_flag': False,
            'pred_state_success_flag': False,
            'pred_new_state': None,
            'experience': experience,
            'compilation_error': f'The predicted new state is the wrong type: {type(pred_new_state)} instead of {type(new_state)}',
        }

    # New state post_step
    code_to_run = "pred_new_state.step()"
    _exec_globals = eval_code(code_to_run, exec_globals=exec_globals, return_exec_globals=True)
    assert not isinstance(_exec_globals, str), f'Failed to run the post_step of the new state: {_exec_globals}'

    # Check the success flag
    pred_state_success_flag = are_two_obj_lists_equal(pred_new_state, new_state)
    success_flag = pred_state_success_flag
    return {
        'success_flag': success_flag,
        'pred_new_state': pred_new_state,
        'pred_state_success_flag': pred_state_success_flag,
        'experience': experience,
        'compilation_error': None
    }

def get_reward_func(code):
    compilation_error, func_name, reward_func = None, None, None
    exec_globals = {}
    exec_globals = eval_code(code, exec_globals=exec_globals, return_exec_globals=True)
    if not isinstance(exec_globals, dict):
        compilation_error = exec_globals
        return compilation_error, func_name, reward_func, exec_globals

    if 'reward_func' in exec_globals:
        func_name = {'reward_func': exec_globals['reward_func']}
    else:
        func_name = {k:v for k, v in exec_globals.items() if not k.startswith('_') and callable(v)}
        if len(func_name) == 0:
            compilation_error = 'No transition function found'
            return compilation_error, func_name, reward_func, exec_globals
        elif len(func_name) > 1:
            tmp_func_name = {k:v for k, v in func_name.items() if 'reward' in k}
            if len(tmp_func_name) >= 1:
                func_name = tmp_func_name
        if len(func_name) > 1:
            print(f'Warning: Expect only one reward function, but got {len(func_name)}')
            lastest_k = list(func_name.keys())[-1]
            func_name = {lastest_k: func_name[lastest_k]}
    func_name = list(func_name.keys())[0]
    reward_func = exec_globals[func_name]
    assert callable(reward_func), f'Expect {func_name} to be callable, but got {reward_func}'
    return compilation_error, func_name, reward_func, exec_globals
def evaluate_reward_code(code, experiences):
    compilation_error, func_name, reward_func, exec_globals = get_reward_func(code)
    if compilation_error is not None:
        results = {
            'success_flag': False,
            'success_ratio': 0,
            'compilation_error': compilation_error,
            'crt_experiences': dict(),
            'wrong_experiences': experiences,
            'experiences': experiences,
            'result_list': [{
                'success_flag': False,
                'pred_reward_success_flag': False,
                'pred_done_success_flag': False,
                'pred_reward': None,
                'pred_done': None,
                'experience': exp,
                'compilation_error': compilation_error,
            } for exp in experiences.values()],
            'func_name': None,
        }
        return results

    result_list = [eval_reward_per_experience(reward_func, exp, exec_globals,) for exp in experiences.values()]
    success_flag = all([result['success_flag'] for result in result_list])
    success_ratio = sum([result['success_flag'] for result in result_list]) / len(result_list)
    pred_reward_success_flag = all([result['pred_reward_success_flag'] for result in result_list])
    pred_reward_success_ratio = sum([result['pred_reward_success_flag'] for result in result_list]) / len(result_list)
    pred_done_success_flag = all([result['pred_done_success_flag'] for result in result_list])
    pred_done_success_ratio = sum([result['pred_done_success_flag'] for result in result_list]) / len(result_list)
    crt_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['success_flag']}
    wrong_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['success_flag']}
    results = {
        'success_flag': success_flag,
        'success_ratio': success_ratio,
        'pred_reward_success_flag': pred_reward_success_flag,
        'pred_reward_success_ratio': pred_reward_success_ratio,
        'pred_done_success_flag': pred_done_success_flag,
        'pred_done_success_ratio': pred_done_success_ratio,
        'compilation_error': None,
        'crt_experiences': crt_experiences,
        'wrong_experiences': wrong_experiences,
        'experiences': experiences,
        'result_list': result_list,
        'func_name': func_name,
    }
    return results
def eval_reward_per_experience(reward_func, experience, exec_globals):
    assert isinstance(experience, dict), f'Expect experience to be a dict, but got {experience}'
    assert 'state' in experience, f'Expect experience to have key "state", but got {list(experience.keys())}'
    assert 'state_next' in experience, f'Expect experience to have key "state_next", but got {list(experience.keys())}'
    assert isinstance(experience['state'], _State), f'Expect experience["state"] to be an instance of _State, but got {experience["state"]}'
    assert isinstance(experience['state_next'], _State), f'Expect experience["state_next"] to be an instance of _State, but got {experience["state_next"]}'
    assert 'action' in experience, f'Expect experience to have key "action", but got {list(experience.keys())}'
    assert isinstance(experience['action'], _Action), f'Expect experience["action"] to be an instance of _Action, but got {experience["action"]}'
    assert 'reward' in experience, f'Expect experience to have key "reward", but got {list(experience.keys())}'
    assert 'done' in experience, f'Expect experience to have key "done", but got {list(experience.keys())}'

    code_to_run = "old_state = experience['state'].to_pyrunnable(exec_globals=exec_globals)"
    _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
    if isinstance(_exec_globals, str):
        pred_reward = None
        pred_done = None
        pred_reward_success_flag = False
        pred_done_success_flag = False
        return {
            'success_flag': False,
            'pred_reward_success_flag': False,
            'pred_done_success_flag': False,
            'pred_reward': pred_reward,
            'pred_done': pred_done,
        }
    old_state = _exec_globals['old_state']

    code_to_run = "new_state = experience['state_next'].to_pyrunnable(exec_globals=exec_globals)"
    _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
    if isinstance(_exec_globals, str):
        pred_reward = None
        pred_done = None
        pred_reward_success_flag = False
        pred_done_success_flag = False
        return {
            'success_flag': False,
            'pred_reward_success_flag': False,
            'pred_done_success_flag': False,
            'pred_reward': pred_reward,
            'pred_done': pred_done,
        }
    new_state = _exec_globals['new_state']

    code_to_run = "action = experience['action'].to_pyrunnable(exec_globals=exec_globals)"
    _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
    if isinstance(_exec_globals, str):
        pred_reward = None
        pred_done = None
        pred_reward_success_flag = False
        pred_done_success_flag = False
        return {
            'success_flag': False,
            'pred_reward_success_flag': False,
            'pred_done_success_flag': False,
            'pred_reward': pred_reward,
            'pred_done': pred_done,
        }
    action = _exec_globals['action']
    reward = experience['reward']
    done = experience['done']

    copied_old_state = copy.deepcopy(old_state)
    copied_action = copy.deepcopy(action)
    copied_new_state = copy.deepcopy(new_state)
    exec_globals.update({
        'copied_old_state': copied_old_state,
        'copied_action': copied_action,
        'copied_new_state': copied_new_state,
        'reward_func': reward_func,
    })
    code_to_exec = 'pred_reward, pred_done = reward_func(copied_old_state, copied_action, copied_new_state)'
    exec_globals = eval_code(code_to_exec, exec_globals=exec_globals, return_exec_globals=True)
    if isinstance(exec_globals, str):
        pred_reward = None
        pred_done = None
        pred_reward_success_flag = False
        pred_done_success_flag = False
        success_flag = False
        compilation_error = exec_globals
    else:
        compilation_error = None
        pred_reward = exec_globals['pred_reward']
        try:
            pred_reward_success_flag = abs(reward - pred_reward) < 1e-6
        except Exception as e:
            pred_reward_success_flag = False
            compilation_error = f'Failed to compare the reward with the following error: {e}'
        pred_done = exec_globals['pred_done']
        pred_done_success_flag = (done == pred_done)
        success_flag = pred_reward_success_flag and pred_done_success_flag
    result = {
        'success_flag': success_flag,
        'pred_reward_success_flag': pred_reward_success_flag,
        'pred_done_success_flag': pred_done_success_flag,
        'pred_reward': pred_reward,
        'pred_done': pred_done,
        'experience': experience,
        'compilation_error': compilation_error,
    }
    return result
