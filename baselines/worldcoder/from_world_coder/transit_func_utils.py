#!/usr/bin/env python
# coding=utf-8

import copy

from .evaluator import evaluate_transit_code

def experiences2text(c):
    old_state_list = [x.input_state.get_str_w_ints_w_touching() for x in c]
    action_list = [x.event for x in c]
    new_state_list = [x.output_state.get_str_w_ints_w_touching() for x in c]
    # difference_list = [text_difference(x.output_state.get_str_w_ints_w_touching(), x.input_state.get_str_w_ints_w_touching()) for x in c]
    text_experiences = [
        # f'The action "{action}" transforms the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n"'.strip()
        # for old_state, action, new_state, diff in zip(old_state_list, action_list, new_state_list, difference_list)
        f'The action "{action}" transforms the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\n'.strip()
        for old_state, action, new_state in zip(old_state_list, action_list, new_state_list,)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences

def experiences2text_with_wrong_outputs(c, code):
    result = evaluate_transit_code(code, c)
    result_list = result['result_list']
    pred_new_state_list = [res['pred_new_state'].get_str_w_ints_w_touching() if res['pred_new_state'] is not None else None for res in result_list]

    old_state_list = [x.input_state.get_str_w_ints_w_touching() for x in c]
    action_list = [x.event for x in c]
    new_state_list = [x.output_state.get_str_w_ints_w_touching() for x in c]
    # difference_list = [text_difference(x['state_next'].get_str_w_ints_w_touching(), x['state'].get_str_w_ints_w_touching()) for x in c]
    compilation_error_list = [res['compilation_error'] for res in result_list]
    # assert len(old_state_list) == len(new_state_list) == len(difference_list) == len(pred_new_state_list) == len(action_list), f'Expect all lists to have the same length, but got {len(old_state_list)}, {len(new_state_list)}, {len(difference_list)}, {len(pred_new_state_list)}, {len(action_list)}'
    assert len(old_state_list) == len(new_state_list) == len(pred_new_state_list) == len(action_list), f'Expect all lists to have the same length, but got {len(old_state_list)}, {len(new_state_list)}, {len(pred_new_state_list)}, {len(action_list)}'

    text_experiences = [
        # _exp2text_wrong(old_state, action, new_state, diff, pred_new_state, compilation_error)
        # for old_state, action, new_state, diff, pred_new_state, compilation_error in zip(old_state_list, action_list, new_state_list, difference_list, pred_new_state_list, compilation_error_list)
        _exp2text_wrong(old_state, action, new_state, pred_new_state, compilation_error)
        for old_state, action, new_state, pred_new_state, compilation_error in zip(old_state_list, action_list, new_state_list, pred_new_state_list, compilation_error_list)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences

def _exp2text_wrong(old_state, action, new_state, pred_new_state, compilation_error):
    if compilation_error is not None:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nHowever, the implementation fails to compile with error\n```\n{compilation_error[:compilation_error.rfind("Printed outputs:")].strip()}\n```\n'.strip()
    elif pred_new_state != new_state:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nHowever, the implementation is wrong because it returns state as \n```\n{pred_new_state}\n```\n'.strip()
    else:
        print(f'Expect the implementation to be wrong, but got correct implementation')
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\n'.strip()
        raise ValueError(f'Expect the implementation to be wrong, but got correct implementation')

def _exp2text_wrong_with_diff(old_state, action, new_state, diff, pred_new_state, compilation_error):
    if compilation_error is not None:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\nHowever, the implementation fails to compile with error\n```\n{compilation_error[:compilation_error.rfind("Printed outputs:")].strip()}\n```\n'.strip()
    elif pred_new_state != new_state:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\nHowever, the implementation is wrong because it returns state as \n```\n{pred_new_state}\n```\n'.strip()
    else:
        print(f'Expect the implementation to be wrong, but got correct implementation')
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n'.strip()
        raise ValueError(f'Expect the implementation to be wrong, but got correct implementation')


