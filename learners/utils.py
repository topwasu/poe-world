import asyncio
import string
import logging
from typing import List, Awaitable, Any

log = logging.getLogger('main')


def list_to_str(lst):
    res = ''
    for idx, x in enumerate(lst):
        res += f'{idx + 1}. {x}\n'
    return res


def list_to_bullets(lst):
    res = ''
    for idx, x in enumerate(lst):
        res += f'- {x}\n'
    return res


def get_2f_list(lst):
    return ' '.join([f"({x:.2f})" for x in lst])


def parse_listed_output(outputs):
    try:
        idx = 1
        res = []
        for output in list(filter(None, outputs.split('\n'))):
            if len(output.split(f'{idx}. ')) > 1:
                res.append(output.split(f'{idx}. ')[1].strip(' \n'))
                idx += 1
        return res
    except:
        log.warning("SOMETHING WRONG", outputs,
                    list(filter(None, outputs.split('\n'))))


def partial_format(prompt, **kwargs):
    keys = [t[1] for t in string.Formatter.parse("", prompt)]
    kwargs = kwargs.copy()
    for key in keys:
        if key not in kwargs and key is not None:
            kwargs[key] = '{' + key + '}'
    return prompt.format(**kwargs)


def process_llm_response_to_codes(x: str) -> List[str]:
    codes = []
    while x.find('```python\n') != -1:
        x = x[x.find('```python\n') + len('```python\n'):]
        codes.append(x[:x.find('```')].strip('\t\n '))
        x = x[x.find('```') + len('```'):]
    return codes


def format_obj_list_w_interaction(obj_list):
    interactions = obj_list.get_obj_interactions()
    txt1 = f'{obj_list}' if len(obj_list) > 0 else 'No objects'
    txt2 = ',\n'.join([repr(xx) for xx in interactions
                       ]) if len(interactions) > 0 else 'No interactions'
    input_obs = txt1 + ',\n' + txt2
    return input_obs


async def await_gather(input_list: List[Awaitable[Any]]) -> List[Any]:
    res = await asyncio.gather(*input_list)
    return res
