#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import copy
import time
import hashlib
import dill
import matplotlib.pyplot as plt

from .init_transit import init_transit
from .refine_transit import refine_transit
from .init_reward import init_reward
from .refine_reward import refine_reward
from .guess_reward import guess_reward
from .refine_to_plan import refine_to_plan

from .evaluator import SepPlanEvaluator, JointEvaluator
from .utils import count_tokens_for_openai, pickable, itemnum, remove_unused_code

def filter_w_missions(experiences, key_missions):
    return SepPlanEvaluator._filter_w_missions(experiences, key_missions)
joint_evaluator = JointEvaluator()
def evaluate(transit_code, reward_code, experiences, envs_to_plan, key_missions, llm, plan_obj_flag=False,):
    for mission in key_missions:
        if mission not in reward_code:
            reward_code[mission] = guess_reward(reward_code, mission, llm, code_example_num=3)
    return joint_evaluator(transit_code, reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag=plan_obj_flag,)
def hashdict(d):
    return tuple(sorted(d.items(), key=str,))
def inverse_hashdict(t):
    return dict(t)

class _Action:
    def __init__(self, strategy, np_rng, parent=None,):
        self.strategy = strategy.strip().lower()
        self.np_rng = np_rng
        self.parent = parent
        assert isinstance(self.np_rng, np.random.Generator), (self, self.np_rng)
        self.local_np_rng = np.random.default_rng(seed=0)
    def init_param(self, *args, **kwargs):
        raise NotImplementedError
    def sample(self,):
        if self.strategy == 'bandits':
            return self.np_rng.beta(self.alpha, self.beta)
        elif self.strategy.startswith('bfs'):
            branching_factor = int(self.strategy.strip('bfs'))
            if self.called_times >= branching_factor:
                return -1e10
            return -self.node_idx
        elif self.strategy.startswith('grid'):
            if self.__class__.__name__.lower().strip().startswith('refine'):
                branching_factor = 1
            elif self.__class__.__name__.lower().strip().startswith('init'):
                branching_factor = int(self.strategy.strip('grid'))
            else:
                raise NotImplementedError
            if self.called_times >= branching_factor:
                return -1e10
            return -self.node_idx
        elif self.strategy.startswith('greedy'):
            return self.value + self.np_rng.normal(0, 1e-5) # Randomness to break ties, for better exploration
        else:
            raise NotImplementedError
    def update_param(self, reward):
        self.called_times += 1
        if self.strategy == 'bandits':
            self.alpha += reward
            self.beta += 1 - reward
            assert self.alpha >= 0 and self.beta >= 0
        elif self.strategy.startswith('bfs'):
            pass
        elif self.strategy.startswith('grid'):
            pass
        elif self.strategy.startswith('greedy'):
            pass
        else:
            raise NotImplementedError


    def hashable(self,):
        raise NotImplementedError
    def __str__(self,):
        raise NotImplementedError
    def __call__(self,):
        raise NotImplementedError

    def __hash__(self,):
        return hash(self.hashable())
    def __eq__(self, other):
        return self.hashable() == other.hashable()
    def __ne__(self, other):
        return self.hashable() != other.hashable()
    def __repr__(self,):
        return str(self)
    def pp(self,):
        return str(self)

class InitAction(_Action):
    def __init__(
        self, node_idx,
        experiences, envs_to_plan, key_missions,
        env_metadata, llm,
        strategy='bandits',
        plan_obj_flag=False,
        experience_num_per_init=7,
        np_rng=None, verbose=False,
    ):
        self.experiences = experiences
        self.envs_to_plan = envs_to_plan
        self.key_missions = key_missions
        self.env_metadata = env_metadata
        self.llm = llm
        self.plan_obj_flag = plan_obj_flag
        self.experience_num_per_init = experience_num_per_init
        self.verbose = verbose

        super().__init__(
            strategy=strategy,
            np_rng=np_rng,
            parent=None,
        )
        self.init_param(node_idx)
    def hashable(self,):
        return (
            'init',
            tuple(sorted(self.experiences, key=str,)),
            tuple(sorted(self.envs_to_plan, key=str,)),
            tuple(sorted(self.key_missions, key=str,)),
            self.experience_num_per_init,
            str(sorted(self.env_metadata.items())),
            self.plan_obj_flag,
        )
    def __str__(self,):
        return f'InitAction(experiences of len {len(self.experiences)} and {len(filter_w_missions(self.experiences, self.key_missions))} for missions {self.key_missions})'
    def __call__(self):
        # print(f'InitAction called {self.called_times} times')
        keys = list(sorted(self.experiences.keys()))
        key_indexes = self.local_np_rng.choice(len(keys), size=min(self.experience_num_per_init, len(keys)), replace=False,)
        experiences = {keys[i]: self.experiences[keys[i]] for i in key_indexes}
        output = init_transit(
            experiences=experiences,
            env_metadata=self.env_metadata,
            llm=self.llm,
            verbose=self.verbose,
        )
        costs = output['final_outputs']['costs']
        transit_code = output['code']

        missions = {str(v['mission']) for v in self.experiences.values()}
        missions = missions.intersection(self.key_missions)
        assert len(missions) > 0, f'no mission in {self.key_missions}'
        mission = sorted(list(missions), key=str)[self.local_np_rng.choice(len(missions))]
        experiences = {k: v for k, v in self.experiences.items() if str(v['mission']) == str(mission)}
        keys = list(sorted(experiences.keys()))
        key_indexes = self.local_np_rng.choice(len(keys), size=min(self.experience_num_per_init, len(keys)), replace=False,)
        experiences = {keys[i]: experiences[keys[i]] for i in key_indexes}
        output = init_reward(
            experiences=experiences,
            env_metadata=self.env_metadata,
            llm=self.llm,
            verbose=self.verbose,
        )
        for k,v in costs.items():
            costs[k] += v
        reward_code = output['code']
        output.update(evaluate(transit_code, reward_code, self.experiences, self.envs_to_plan, self.key_missions, self.plan_obj_flag, self.llm,))
        output['transit_code'] = transit_code
        output['reward_code'] = reward_code
        output['final_outputs']['costs'] = costs

        print(f'InitAction called {self.called_times} times')
        self.update_param(output['success_flag'])
        return output

    def init_param(self, node_idx):
        self.called_times = 0
        if self.strategy == 'bandits':
            self.alpha = 1
            self.beta = 1
        elif self.strategy.startswith('bfs'):
            self.node_idx = node_idx
        elif self.strategy.startswith('grid'):
            self.node_idx = node_idx
        elif self.strategy.startswith('greedy'):
            self.value = 0
        else:
            raise NotImplementedError

class RefineTransitAction(_Action):
    def __init__(
        self, node_idx,
        init_transit_code, init_reward_code,
        experiences, envs_to_plan, key_missions,
        llm,
        strategy='bandits',
        plan_obj_flag=False,
        eval_result=None,
        bandits_C=5,
        alpha=None, beta=None, parent=None,
        crt_exp_num_per_refine=3,
        np_rng=None, verbose=False,
    ):
        assert parent is not None, f'{self.__class__.__name__} must have a parent'
        self.init_transit_code = init_transit_code
        self.init_reward_code = init_reward_code
        self.experiences = experiences
        self.envs_to_plan = envs_to_plan
        self.key_missions = key_missions
        self.llm = llm
        self.plan_obj_flag = plan_obj_flag
        self.bandits_C = bandits_C
        self.crt_exp_num_per_refine = crt_exp_num_per_refine
        self.verbose = verbose

        if eval_result is None:
            eval_result = evaluate(init_transit_code, init_reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag, self.llm,)
        self.eval_result = eval_result
        # assert len(self.eval_result['exp_result']['crt_transit_experiences']) > 0, f'len(crt_experiences) == 0'
        # assert len(self.eval_result['exp_result']['wrong_transit_experiences']) > 0, (f'len(wrong_experiences) == 0', {k:v for k,v in self.eval_result['exp_result'].items() if len(str(v)) < 1000})

        super().__init__(
            strategy=strategy,
            np_rng=np_rng,
            parent=parent,
        )
        self.init_param(node_idx, eval_result)

    def hashable(self,):
        return (
            self.__class__.__name__, self.init_transit_code, hashdict(self.init_reward_code),
            tuple(sorted(self.experiences.keys(), key=str,)), tuple(sorted(self.envs_to_plan, key=str,)),
            tuple(sorted(self.key_missions, key=str,)),
            self.crt_exp_num_per_refine,
            self.plan_obj_flag,
        )
    def __str__(self,):
        return f'{self.__class__.__name__}(init code of exp success ratio {self.eval_result["exp_result"]["success_ratio"]}, plan success ratio {self.eval_result["plan_result"]["success_ratio"]})'

    def __call__(self,):
        output = self.main_call()
        print(f'{self.__class__.__name__} called {self.called_times} times')
        self.update_param(output['success_flag'])
        return output
    def main_call(self,):
        exp_result = self.eval_result['exp_result']
        crt_experiences = exp_result['crt_transit_experiences']
        wrong_experiences = exp_result['wrong_transit_experiences']
        assert len(crt_experiences) > 0, f'len(crt_experiences) == 0'
        assert len(wrong_experiences) > 0, f'len(wrong_experiences) == 0'
        if len(crt_experiences) > self.crt_exp_num_per_refine:
            crt_keys = list(sorted(crt_experiences.keys()))
            crt_key_indexes = self.local_np_rng.choice(len(crt_keys), size=self.crt_exp_num_per_refine, replace=False)
            crt_experiences = {crt_keys[i]: crt_experiences[crt_keys[i]] for i in crt_key_indexes}
        else:
            crt_experiences = crt_experiences
        wrong_keys = list(sorted(wrong_experiences.keys()))
        wrong_key_indexes = self.local_np_rng.choice(len(wrong_keys), size=1, replace=False)
        wrong_experiences = {wrong_keys[i]: wrong_experiences[wrong_keys[i]] for i in wrong_key_indexes}
        output = refine_transit(
            code=self.init_transit_code,
            crt_experiences=crt_experiences,
            wrong_experiences=wrong_experiences,
            llm=self.llm,
            verbose=self.verbose,
        )
        output.update(evaluate(output['code'], self.init_reward_code, self.experiences, self.envs_to_plan, self.key_missions, self.plan_obj_flag, self.llm,))
        output['transit_code'] = output['code']
        output['reward_code'] = self.init_reward_code
        return output

    def init_param(self, node_idx, eval_result):
        self.called_times = 0
        if self.strategy == 'bandits':
            self.alpha = 1 + eval_result['success_ratio'] * self.bandits_C
            self.beta = 1 + (1 - eval_result['success_ratio']) * self.bandits_C
        elif self.strategy.startswith('bfs'):
            self.node_idx = node_idx
        elif self.strategy.startswith('grid'):
            self.node_idx = node_idx
        elif self.strategy.startswith('greedy'):
            self.value = eval_result['success_ratio']
        else:
            raise NotImplementedError

class RefineRewardAction(RefineTransitAction):
    def main_call(self,):
        exp_result = self.eval_result['exp_result']
        crt_experiences = exp_result['crt_reward_experiences']
        wrong_experiences = exp_result['wrong_reward_experiences']
        # assert len(crt_experiences) > 0, f'len(crt_experiences) == 0'
        assert len(wrong_experiences) > 0, f'len(wrong_experiences) == 0'
        missions_with_crt_exp = {str(v['mission']) for v in crt_experiences.values()}
        missions = {str(v['mission']) for v in wrong_experiences.values() if str(v['mission']) in missions_with_crt_exp}
        missions = missions.intersection(self.key_missions)
        if len(missions) == 0:
            missions = {str(v['mission']) for v in wrong_experiences.values()}
            missions = missions.intersection(self.key_missions)
        assert len(missions) > 0, f'len(missions) == 0'
        mission = sorted(list(missions), key=str)[self.local_np_rng.choice(len(missions))]
        crt_experiences = {k: v for k, v in crt_experiences.items() if str(v['mission']) == mission}
        wrong_experiences = {k: v for k, v in wrong_experiences.items() if str(v['mission']) == mission}
        # assert len(crt_experiences) > 0, f'len(crt_experiences) == 0'
        assert len(wrong_experiences) > 0, f'len(wrong_experiences) == 0'
        if len(crt_experiences) > self.crt_exp_num_per_refine:
            crt_keys = list(sorted(crt_experiences.keys()))
            crt_key_indexes = self.local_np_rng.choice(len(crt_keys), size=min(self.crt_exp_num_per_refine, len(crt_keys)), replace=False)
            crt_experiences = {crt_keys[i]: crt_experiences[crt_keys[i]] for i in crt_key_indexes}
        else:
            crt_experiences = crt_experiences
        wrong_keys = list(sorted(wrong_experiences.keys()))
        wrong_key_indexes = self.local_np_rng.choice(len(wrong_keys), size=1, replace=False)
        wrong_experiences = {wrong_keys[i]: wrong_experiences[wrong_keys[i]] for i in wrong_key_indexes}
        output = refine_reward(
            lambda_code=self.init_reward_code,
            crt_experiences=crt_experiences,
            wrong_experiences=wrong_experiences,
            llm=self.llm,
            verbose=self.verbose,
        )
        output.update(evaluate(self.init_transit_code, output['code'], self.experiences, self.envs_to_plan, self.key_missions, self.plan_obj_flag, self.llm,))
        output['transit_code'] = self.init_transit_code
        output['reward_code'] = output['code']
        return output

class RefineToPlanTRAction(RefineTransitAction):
    def main_call(self,):
        plan_result = self.eval_result['plan_result']
        wrong_envs = plan_result['wrong_envs']
        wrong_keys = list(sorted(wrong_envs.keys()))
        wrong_key_indexes = self.local_np_rng.choice(len(wrong_keys), size=1, replace=False)[0]
        wrong_env = wrong_envs[wrong_keys[wrong_key_indexes]]
        output = refine_to_plan(
            transit_code=self.init_transit_code,
            reward_code=self.init_reward_code[str(wrong_env['mission'])],
            failed_env_info = wrong_env,
            llm=self.llm,
            verbose=self.verbose,
        )
        transit_code = remove_unused_code(output['code'], 'transition')
        reward_code = remove_unused_code(output['code'], 'reward_func')
        lambda_reward_code = copy.deepcopy(self.init_reward_code)
        lambda_reward_code[str(wrong_env['mission'])] = reward_code
        output.update(evaluate(transit_code, lambda_reward_code, self.experiences, self.envs_to_plan, self.key_missions, self.plan_obj_flag, self.llm,))
        output['transit_code'] = transit_code
        output['reward_code'] = lambda_reward_code
        return output

def synthesis(
    experiences, envs_to_plan, key_missions,
    init_transit_code=None,
    init_reward_code=None,
    env_metadata=None, llm=None,
    strategy='bandits',
    verbose=False,
    max_step_num=300, budget=1000000,
    plan_obj_flag=False,
    bandits_C=5.0,
    np_rng=None,
):
    assert init_reward_code is None or isinstance(init_reward_code, dict)
    key_missions = set([str(m) for m in key_missions])
    if np_rng is None:
        np_rng = np.random.default_rng(seed=0)
    elif isinstance(np_rng, int):
        np_rng = np.random.default_rng(seed=np_rng)
    assert isinstance(np_rng, np.random.Generator)
    transit_cache_dir = None
    reward_cache_dir = None
    print('~' * 20, 'synthesis_plan_by_bandits', '~' * 20)
    total_costs = None
    total_new_costs = None

    curdir = os.path.dirname(os.path.abspath(__file__))
    cachedir = os.path.join(curdir, 'cache', 'synthesis_plan_by_bandits')
    os.makedirs(cachedir, exist_ok=True)
    exp_id = hashlib.md5(str((
        tuple(sorted(experiences.keys(), key=str,)),
        tuple(sorted(envs_to_plan, key=str,)),
        tuple(sorted(key_missions)),
        str(sorted(env_metadata.items())), str(sorted(llm.default_args.items())),
        plan_obj_flag,
        strategy,
        bandits_C,
    )).encode()).hexdigest()
    cache_dir = os.path.join(cachedir, exp_id)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cachedir, f'{exp_id}.pkl')
    # set_logger(os.path.join(cache_dir, f'{exp_id}.log'))

    loaded_cache_flag = False
    if os.path.exists(cache_path):
        print(f'Found cache at {cache_path}, but skip loading')
    else:
        print(f'cache not found at {cache_path}, start from scratch')
    if not loaded_cache_flag:
        logs_history = []
        logs = {
            'step': 0,
            'logs_history': logs_history,
            'transit_cache_dir': transit_cache_dir,
            'reward_cache_dir': reward_cache_dir,
            'configurations': {
                'experiences': experiences,
                'envs_to_plan': envs_to_plan,
                'key_missions': key_missions,
                'init_transit_code': init_transit_code,
                'init_reward_code': init_reward_code,
                'env_metadata': env_metadata,
                'llm_default_args': llm.default_args,
                'plan_obj_flag': plan_obj_flag,
                'strategy': strategy,
                'bandits_C': bandits_C,
                'verbose': verbose,
                'max_step_num': max_step_num,
                'budget': budget,
            },
        }

    if 'actions' not in logs:
        actions = [
            InitAction(
                0,
                experiences, envs_to_plan, key_missions,
                env_metadata, llm,
                strategy=strategy,
                plan_obj_flag=plan_obj_flag,
                np_rng=np_rng, verbose=verbose,
            ),
        ]
        assert actions[0] == actions[0], 'InitAction must be hashable'
        visited_actions = set(actions)
        best_success_ratio, best_outputs = (0, 0), []
        exp_success_ratio_list = []
        best_exp_success_ratio_list = []
        node_idx = 0
        code2node_idx = {}
        perfect_transit_code = set()
        perfect_reward_code = set()

        if init_transit_code is not None and init_reward_code is not None:
            eval_result = evaluate(
                init_transit_code, init_reward_code,
                experiences, envs_to_plan, key_missions,
                plan_obj_flag=plan_obj_flag,
                llm=llm,
            )
            if eval_result['success_flag']:
                eval_result['total_costs'] = {
                    'total_tokens': [],
                    'prompt_tokens': [],
                    'completion_tokens': [],
                    'requests': [],
                }
                eval_result['total_new_costs'] = {
                    'total_tokens': [],
                    'prompt_tokens': [],
                    'completion_tokens': [],
                    'requests': [],
                }
                print('Found perfect solution in the beginning')
                return eval_result, cache_dir
            if eval_result['exp_result']['transit_success_flag']:
                perfect_transit_code.add(init_transit_code)
            if eval_result['exp_result']['reward_success_flag']:
                perfect_reward_code.add(hashdict(init_reward_code))
            best_success_ratio = (eval_result['exp_result']['success_ratio'], eval_result['plan_result']['success_ratio'] if eval_result['plan_result']['success_ratio'] is not None else 0)
            best_outputs.append(eval_result)

            code2node_idx[(init_transit_code, hashdict(init_reward_code))] = node_idx
            node_idx += 1
            exp_success_ratio_list.append(eval_result['exp_result']['success_ratio'])
            best_exp_success_ratio_list.append(max(exp_success_ratio_list))

            if not eval_result['exp_result']['transit_success_flag']:
                actions.append(RefineTransitAction(
                    len(actions),
                    init_transit_code,
                    init_reward_code,
                    experiences, envs_to_plan, key_missions,
                    llm,
                    strategy=strategy,
                    eval_result=eval_result,
                    plan_obj_flag=plan_obj_flag,
                    bandits_C=bandits_C,
                    np_rng=np_rng,
                    verbose=verbose,
                    parent='FAKE',
                ))
                visited_actions.add(actions[-1])
            # if eval_result['exp_result']['transit_success_flag'] and \
            if True and \
                    not eval_result['exp_result']['reward_success_flag']:
                actions.append(RefineRewardAction(
                    len(actions),
                    init_transit_code,
                    init_reward_code,
                    experiences, envs_to_plan, key_missions,
                    llm,
                    eval_result=eval_result,
                    strategy=strategy,
                    plan_obj_flag=plan_obj_flag,
                    bandits_C=bandits_C,
                    np_rng=np_rng,
                    verbose=verbose,
                    parent='FAKE',
                ))
                visited_actions.add(actions[-1])
            if plan_obj_flag and (
                eval_result['plan_result'] is not None and
                not eval_result['plan_result']['success_flag']
            ) and eval_result['exp_result']['success_flag']:
                actions.append(RefineToPlanTRAction(
                    len(actions),
                    init_transit_code, init_reward_code,
                    experiences, envs_to_plan, key_missions,
                    llm,
                    eval_result=eval_result,
                    strategy=strategy,
                    plan_obj_flag=plan_obj_flag,
                    bandits_C=bandits_C,
                    np_rng=np_rng,
                    verbose=verbose,
                    parent='FAKE',
                ))
                visited_actions.add(actions[-1])

        logs['actions'] = actions
        logs['visited_actions'] = visited_actions
        logs['best_success_ratio'] = best_success_ratio
        logs['best_outputs'] = best_outputs
        logs['exp_success_ratio_list'] = exp_success_ratio_list
        logs['best_exp_success_ratio_list'] = best_exp_success_ratio_list
        logs['node_idx'] = node_idx
        logs['code2node_idx'] = code2node_idx
        logs['perfect_transit_code'] = perfect_transit_code
        logs['perfect_reward_code'] = perfect_reward_code
        with open(cache_path, 'wb') as f:
            dill.dump(logs, f)
    else:
        actions = logs['actions']
        visited_actions = logs['visited_actions']
        best_success_ratio = logs['best_success_ratio']
        best_outputs = logs['best_outputs']
        exp_success_ratio_list = logs['exp_success_ratio_list']
        best_exp_success_ratio_list = logs['best_exp_success_ratio_list']
        node_idx = logs['node_idx']
        code2node_idx = logs['code2node_idx']
        perfect_transit_code = logs['perfect_transit_code']
        perfect_reward_code = logs['perfect_reward_code']

    start_time = time.time()
    for step in range(logs['step'], max_step_num):
        if time.time() - start_time > budget:
            break
        print('='*10, f'step {step}', '='*10)

        sampled_theta_list = [action.sample() for action in actions]
        action_index = np.argmax(sampled_theta_list)
        action = actions[action_index]

        print('='*10, action, '='*10)
        output = action()
        costs = output['final_outputs']['costs']
        if total_costs is None:
            total_costs = {k:[v] for k,v in costs.items()}
        else:
            for k,v in costs.items():
                total_costs[k].append(v)
        new_costs = output['final_outputs']['new_costs']
        if total_new_costs is None:
            total_new_costs = {k:[v] for k,v in new_costs.items()}
        else:
            for k,v in new_costs.items():
                total_new_costs[k].append(v)
        # print('mean costs:', {k: np.mean(v) for k, v in total_costs.items() if len(v) > 0})
        print('total costs:', {k: np.sum(v) for k, v in total_costs.items() if len(v) > 0})
        print('total new costs:', {k: np.sum(v) for k, v in total_new_costs.items() if len(v) > 0})
        output['total_costs'] = total_costs
        output['total_new_costs'] = total_new_costs

        exp_result = output['exp_result']
        print('exp_result:', {k: v for k, v in exp_result.items() if itemnum(v) < 50 and len(str(v)) < 100})
        plan_result = output['plan_result']
        if plan_result is not None:
            print('plan_result:', {k: v for k, v in plan_result.items() if itemnum(v) < 50 and len(str(v)) < 100})

        if (output['exp_result']['success_ratio'], output['plan_result']['success_ratio'] if output['plan_result']['success_ratio'] is not None else 0) > best_success_ratio:
            best_success_ratio = (output['exp_result']['success_ratio'], output['plan_result']['success_ratio'] if output['plan_result']['success_ratio'] is not None else 0)
            best_outputs = [output,]
        elif (output['exp_result']['success_ratio'], output['plan_result']['success_ratio'] if output['plan_result']['success_ratio'] is not None else 0) == best_success_ratio:
            best_outputs.append(output)

        if output['exp_result']['transit_success_flag'] and output['transit_code'] not in perfect_transit_code:
            print('found new perfect transit code, do cross-product-evaluation')
            perfect_transit_code.add(output['transit_code'])
            for prc in sorted(list(perfect_reward_code)):
                prc = inverse_hashdict(prc)
                tmp_output = evaluate(output['transit_code'], prc, experiences, envs_to_plan, key_missions, plan_obj_flag=plan_obj_flag, llm=llm)
                tmp_output['total_costs'] = total_costs
                tmp_output['total_new_costs'] = total_new_costs
                if tmp_output['success_flag']:
                    return tmp_output, cache_dir
                if (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0) > best_success_ratio:
                    print('found better output by cross product')
                    best_success_ratio = (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0)
                    best_outputs = [tmp_output,]
                elif (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0) == best_success_ratio:
                    best_outputs.append(tmp_output)
        if output['exp_result']['reward_success_flag'] and hashdict(output['reward_code']) not in perfect_reward_code:
            print('found new perfect reward code, do cross-product-evaluation')
            perfect_reward_code.add(hashdict(output['reward_code']))
            for ptc in sorted(list(perfect_transit_code)):
                tmp_output = evaluate(ptc, output['reward_code'], experiences, envs_to_plan, key_missions, plan_obj_flag=plan_obj_flag, llm=llm,)
                tmp_output['total_costs'] = total_costs
                tmp_output['total_new_costs'] = total_new_costs
                if tmp_output['success_flag']:
                    return tmp_output, cache_dir
                if (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0) > best_success_ratio:
                    print('found better output by cross product')
                    best_success_ratio = (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0)
                    best_outputs = [tmp_output,]
                elif (tmp_output['exp_result']['success_ratio'], tmp_output['plan_result']['success_ratio'] if tmp_output['plan_result']['success_ratio'] is not None else 0) == best_success_ratio:
                    best_outputs.append(tmp_output)

        eval_result = output
        if not eval_result['exp_result']['transit_success_flag'] and len(eval_result['exp_result']['crt_transit_experiences']) > 0:
            new_action = RefineTransitAction(
                len(actions),
                output['transit_code'], output['reward_code'],
                experiences, envs_to_plan, key_missions,
                llm,
                eval_result=eval_result,
                strategy=strategy,
                plan_obj_flag=plan_obj_flag,
                bandits_C=bandits_C,
                np_rng=np_rng,
                verbose=verbose,
                parent=action,
            )
            assert new_action == new_action, 'RefineAction must be hashable'
            if new_action not in visited_actions:
                actions.append(new_action)
                visited_actions.add(new_action)
                print(f'Add new action {new_action}')
        # if eval_result['exp_result']['transit_success_flag'] and \
        if True and \
                not eval_result['exp_result']['reward_success_flag'] and len(eval_result['exp_result']['crt_reward_experiences']) > 0:
            new_action = RefineRewardAction(
                len(actions),
                output['transit_code'], output['reward_code'],
                experiences, envs_to_plan, key_missions,
                llm,
                eval_result=eval_result,
                plan_obj_flag=plan_obj_flag,
                strategy=strategy,
                bandits_C=bandits_C,
                np_rng=np_rng,
                verbose=verbose,
                parent=action,
            )
            assert new_action == new_action, 'RefineAction must be hashable'
            if new_action not in visited_actions:
                actions.append(new_action)
                visited_actions.add(new_action)
                print(f'Add new action {new_action}')

        if plan_obj_flag and (
            eval_result['plan_result'] is not None and
            not eval_result['plan_result']['success_flag']
        ) and eval_result['exp_result']['success_flag']:
            new_action = RefineToPlanTRAction(
                len(actions),
                output['transit_code'], output['reward_code'],
                experiences, envs_to_plan, key_missions,
                llm,
                eval_result=eval_result,
                strategy=strategy,
                plan_obj_flag=plan_obj_flag,
                bandits_C=bandits_C,
                np_rng=np_rng,
                verbose=verbose,
                parent=action,
            )
            assert new_action == new_action, 'RefineToPlanAction must be hashable'
            if new_action not in visited_actions:
                actions.append(new_action)
                visited_actions.add(new_action)
                print(f'Add new action {new_action}')

        if action.parent is None:
            code2node_idx[(output['transit_code'], hashdict(output['reward_code']))] = node_idx
            node_idx += 1
            exp_success_ratio_list.append(output['exp_result']['success_ratio'])
            best_exp_success_ratio_list.append(max(exp_success_ratio_list))
        else:
            code2node_idx[(output['transit_code'], hashdict(output['reward_code']))] = node_idx
            node_idx += 1
            exp_success_ratio_list.append(output['exp_result']['success_ratio'])
            best_exp_success_ratio_list.append(max(exp_success_ratio_list))

        plt.plot(exp_success_ratio_list, label='exp_success_ratio')
        plt.plot(best_exp_success_ratio_list, label='best_exp_success_ratio')
        plt.legend()
        plt.savefig(os.path.join(cache_dir, 'learning_curve.pdf'))
        plt.close()

        start_time = time.time()
        logs['sampled_theta_list'] = sampled_theta_list
        logs['action_index'] = action_index
        logs['action'] = action
        logs['costs'] = costs
        logs['output'] = output
        logs['best_success_ratio'] = best_success_ratio
        logs['best_outputs'] = best_outputs
        logs['actions'] = actions
        logs['visited_actions'] = visited_actions
        logs['step'] = step + 1
        logs['code2node_idx'] = code2node_idx
        logs['node_idx'] = node_idx
        logs['total_costs'] = total_costs
        logs['total_new_costs'] = total_new_costs
        logs_history.append({
            k: v for k, v in logs.items() if k not in ['logs_history',]
        })
        logs['logs_history'] = logs_history
        assert len(logs_history) == logs['step']
        logging_time = time.time() - start_time
        if step % 10 == 0 or step == max_step_num - 1 or output['success_flag']:
            with open(cache_path, 'wb') as f:
                dill.dump(pickable(logs), f)
        save_time = time.time() - start_time - logging_time
        print(f'logging_time: {logging_time:.2f}s, save_time: {save_time:.2f}s')

        with open(os.path.join(cache_dir, f'step{step+1}_logs_{output["exp_result"]["success_ratio"]*100 :.1f}_{action.__class__.__name__}_{action.eval_result["success_ratio"]*100 if "eval_result" in dir(action) else 0 :.1f}.pkl'), 'wb') as f:
            dill.dump(pickable(logs_history[-1]), f)

        if output['success_flag']:
            return output, cache_dir

    print(f'After {step+1} steps, best success ratio is {best_success_ratio} with {len(best_outputs)} possible solutions. Randomly choose one.')
    output = np_rng.choice(best_outputs)
    output['total_costs'] = total_costs
    output['total_new_costs'] = total_new_costs
    return output, cache_dir
