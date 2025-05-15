# coding=utf-8

import os
import copy
from dataclasses import dataclass

from openai import OpenAI #TODO: change it to OpenRouter

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, select, ARRAY, Float
from sqlalchemy.orm import Session

Base = declarative_base()
class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""
    __tablename__ = "full_llm_cache"
    prompt = Column(String, primary_key=True)
    model_args = Column(String, primary_key=True)
    response = Column(String)
    completion_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    total_tokens = Column(Integer)
    idx = Column(Integer, primary_key=True)
ENGINES = {
    seed: create_engine('sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), f'full_llm_cache_{seed}.db'))
    for seed in range(1000)
}

class LLM:
    def __init__(self, default_args={'model': 'gpt-4o', 'temperature': 1.0,}, seed=None,):
        self.llm = _LLM(default_args=copy.deepcopy(default_args), seed=seed,)
        self.tracker = LLMUsageTracker()
    def __call__(self, prompt, model_args=None):
        completion = self.llm(prompt, model_args)
        self.tracker.update(completion)
        return completion
    def track(self, name=None):
        return self.tracker.track(name)
    def track_new(self, name=None):
        return self.llm.track(name)
    @property
    def default_args(self):
        return self.llm.default_args

class _LLM:
    def __init__(self, default_args={'model': 'gpt-4o', 'temperature': 1.0,}, seed=0,):
        assert seed in ENGINES, f"seed: {seed} not in {ENGINES.keys()}"
        self.engine = ENGINES[seed]
        self.default_args = default_args
        self.client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=os.environ['KEVIN_OPEN_ROUTER_KEY'],
        )
        self.local_tracker = LLMUsageTracker()
        self.nth_dict = {}
        self.cache_schema = FullLLMCache
        self.cache_schema.metadata.create_all(self.engine)
    def __call__(self, prompt, model_args=None):
        model_args = self._merge_args(model_args)
        prompt_in_str = '\n\n'.join([p['role']+':\n'+p['content'] for p in prompt])
        model_args_in_str = str(sorted(model_args.items()))
        hash_idx = prompt_in_str + '\n' + model_args_in_str

        # Handle nth
        if hash_idx not in self.nth_dict:
            self.nth_dict[hash_idx] = -1
        self.nth_dict[hash_idx] += 1
        nth = self.nth_dict[hash_idx]

        # Retrieve the cache
        stmt = (
            select(
                self.cache_schema.response,
                self.cache_schema.completion_tokens,
                self.cache_schema.prompt_tokens,
                self.cache_schema.total_tokens,
                self.cache_schema.idx,
            ).where(
                self.cache_schema.prompt == prompt_in_str,
                self.cache_schema.model_args == model_args_in_str,
            ).order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            generations = session.execute(stmt).fetchall()
            generations = [row for row in generations]
            generations.sort(key=lambda x: x[-1])
            if nth < len(generations):
                print('Retrieved from cache, LLM request, nth:', nth)
                return db2fake_completion(generations[nth])

        # Actual LLM request
        completion = self._action(prompt, model_args)
        response, completion_tokens, prompt_tokens, total_tokens = completion2db(completion)
        item = FullLLMCache(
            prompt=prompt_in_str,
            model_args=model_args_in_str,
            response=response,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            idx=nth,
        )
        with Session(self.engine) as session, session.begin():
            session.add(item)
        return completion

    def _action(self, prompt, model_args=None):
        print('new LLM request')
        model_args = self._merge_args(model_args)
        out = self.client.chat.completions.create(messages=prompt, **model_args)
        assert str(out).startswith('ChatCompletion('), f"out: {out}"
        ct = 0
        while ct < 10:
            try:
                self.local_tracker.update(out)
                break
            except Exception as e:
                out = self.client.chat.completions.create(messages=prompt, **model_args)
                assert str(out).startswith('ChatCompletion('), f"out: {out}"
                ct += 1
        if ct == 10:
            raise Exception(f"Failed to update local tracker after 10 retries")
        return out
    def _merge_args(self, model_args):
        args = copy.deepcopy(self.default_args)
        if model_args is not None:
            args.update(model_args)
        return args
    def track(self, name=None):
        return self.local_tracker.track(name)
    def __getstate__(self):
        return {
            'default_args': self.default_args,
            'local_tracker': self.local_tracker,
        }
    def __setstate__(self, state):
        self.default_args = state['default_args']
        self.local_tracker = state['local_tracker']

def completion2db(completion):
    response = completion.choices[0].message.content
    completion_tokens = completion.usage.completion_tokens
    prompt_tokens = completion.usage.prompt_tokens
    total_tokens = completion.usage.total_tokens
    return response, completion_tokens, prompt_tokens, total_tokens
def db2fake_completion(db_row):
    response, completion_tokens, prompt_tokens, total_tokens, _ = db_row
    message = FakeMessage(content=response)
    usage = FakeUsage(completion_tokens=completion_tokens, prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    completion = FakeCompletion(choices=[FakeChoice(message=message)], usage=usage)
    return completion
@dataclass
class FakeMessage:
    content: str
@dataclass
class FakeUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
@dataclass
class FakeCompletion:
    choices: list
    usage: FakeUsage
@dataclass
class FakeChoice:
    message: FakeMessage

class LLMUsageTracker:
    def __init__(self,):
        self.usage = {
            'requests': 0,
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
            'cost': 0,
        }
        self.minor_trackers = {}
    def update(self, completion):
        self.usage['requests'] += 1
        self.usage['completion_tokens'] += completion.usage.completion_tokens
        self.usage['prompt_tokens'] += completion.usage.prompt_tokens
        self.usage['total_tokens'] += completion.usage.total_tokens
        self.usage['cost'] += 2.5e-6 * completion.usage.prompt_tokens + 1e-5 * completion.usage.completion_tokens
    def __str__(self):
        return str(self.usage)
    def __repr__(self):
        return str(self.usage)
    def track(self, name=None):
        if name is None:
            return MinorLLMUsageTracker(self, name)
        if name not in self.minor_trackers:
            self.minor_trackers[name] = MinorLLMUsageTracker(self, name)
        return self.minor_trackers[name]
class MinorLLMUsageTracker():
    def __init__(self, tracker, name):
        self.name = name
        self.tracker = tracker
        self.usage = {
            'requests': 0,
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
            'cost': 0,
        }
    def __enter__(self,):
        self.start = copy.deepcopy(self.tracker.usage)
        return self
    def __exit__(self, type, value, traceback):
        self.end = copy.deepcopy(self.tracker.usage)
        self.usage['requests'] += self.end['requests'] - self.start['requests']
        self.usage['completion_tokens'] += self.end['completion_tokens'] - self.start['completion_tokens']
        self.usage['prompt_tokens'] += self.end['prompt_tokens'] - self.start['prompt_tokens']
        self.usage['total_tokens'] += self.end['total_tokens'] - self.start['total_tokens']
        self.usage['cost'] += self.end['cost'] - self.start['cost']
    def __str__(self):
        return str(self.usage)
    def __repr__(self):
        return str(self.usage)
