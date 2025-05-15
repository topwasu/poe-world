#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))

import json
import dill

from worldcoder.agent.synthesizer.llm_utils import FullLLMCache, ENGINE, Session, completion2db

cache_schema = FullLLMCache
cache_schema.metadata.create_all(ENGINE)

old_cache_path = osp.join(osp.dirname(osp.abspath(__file__)), 'cached_data', '_LLM')
for prompt_id in os.listdir(old_cache_path):
    prompt_path = osp.join(old_cache_path, prompt_id)
    if not osp.isdir(prompt_path):
        continue
    assert osp.isdir(prompt_path), prompt_path
    with open(osp.join(prompt_path, 'value.json')) as f:
        prompt = json.load(f)
    prompt_in_str = '\n\n'.join([p['role']+':\n'+p['content'] for p in prompt])
    for model_args_id in os.listdir(prompt_path):
        model_args_path = osp.join(prompt_path, model_args_id)
        if not osp.isdir(model_args_path):
            continue
        if not osp.exists(osp.join(model_args_path, 'value.json')):
            continue
        if not osp.exists(osp.join(model_args_path, 'cache.dill')):
            continue
        assert osp.isdir(model_args_path), model_args_path
        with open(osp.join(model_args_path, 'value.json')) as f:
            model_args = json.load(f)
        model_args_in_str = str(sorted(model_args.items()))

        with open(os.path.join(model_args_path, 'cache.dill'), 'rb') as f:
            cache = dill.load(f)
        responses = cache['response']
        for nth, completion in enumerate(responses):
            response, completion_tokens, prompt_tokens, total_tokens = completion2db(completion)
            with Session(ENGINE) as session, session.begin():
                session.add(cache_schema(
                    prompt=prompt_in_str,
                    model_args=model_args_in_str,
                    response=response,
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                    idx=nth,
                ))

