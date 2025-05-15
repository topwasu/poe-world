import os
import numpy as np
import logging

from .init_transit import init_transit
from .refine_transit import refine_transit

from .evaluator import evaluate_transit_code
from .utils import count_tokens_for_openai, pickable, itemnum, remove_unused_code

log = logging.getLogger('main')


def evaluate(transit_code, experiences,):
    res = evaluate_transit_code(transit_code, experiences,)
    return {'exp_result': res, 'success_flag': res['success_flag'], 'success_ratio': res['success_ratio'],}

class _Action:
    def __init__(self, np_rng, parent=None,):
        self.np_rng = np_rng
        self.parent = parent
        assert isinstance(self.np_rng, np.random.Generator), (self, self.np_rng)
        # self.local_np_rng = np.random.default_rng(seed=0)
        self.local_np_rng = np_rng
    def init_param(self, *args, **kwargs):
        raise NotImplementedError
    def sample(self,):
        return self.np_rng.beta(self.alpha, self.beta)
    def update_param(self, reward):
        self.alpha += reward
        self.beta += 1 - reward
        assert self.alpha >= 0 and self.beta >= 0


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
        experiences,
        llm,
        experience_num_per_init=7,
        np_rng=None, verbose=False,
    ):
        self.experiences = experiences
        self.llm = llm
        self.experience_num_per_init = experience_num_per_init
        self.verbose = verbose

        super().__init__(
            np_rng=np_rng,
            parent=None,
        )
        self.init_param(node_idx)
    def hashable(self,):
        return (
            'init',
            tuple(sorted(self.experiences, key=str,)),
            self.experience_num_per_init,
        )
    def __str__(self,):
        return f'InitAction(experiences of len {len(self.experiences)})'
    def __call__(self):
        indexes = self.local_np_rng.choice(len(self.experiences), size=min(self.experience_num_per_init, len(self.experiences)), replace=False,)
        experiences = [self.experiences[i] for i in indexes]
        output = init_transit(
            experiences=experiences,
            llm=self.llm,
            verbose=self.verbose,
        )
        costs = output['final_outputs']['costs']
        transit_code = output['code']

        output.update(evaluate(transit_code, self.experiences,))
        output['transit_code'] = transit_code
        output['final_outputs']['costs'] = costs

        log.debug(f'InitAction called {self.called_times} times')
        self.update_param(output['success_flag'])
        return output

    def init_param(self, node_idx):
        self.called_times = 0
        self.alpha = 1
        self.beta = 1

class RefineTransitAction(_Action):
    def __init__(
        self, node_idx,
        init_transit_code,
        experiences,
        llm,
        eval_result=None,
        bandits_C=20,
        alpha=None, beta=None, parent=None,
        crt_exp_num_per_refine=3,
        np_rng=None, verbose=False,
    ):
        assert parent is not None, f'{self.__class__.__name__} must have a parent'
        self.init_transit_code = init_transit_code
        self.experiences = experiences
        self.llm = llm
        self.bandits_C = bandits_C
        self.crt_exp_num_per_refine = crt_exp_num_per_refine
        self.verbose = verbose

        if eval_result is None:
            eval_result = evaluate(init_transit_code, experiences,)
        self.eval_result = eval_result

        super().__init__(
            np_rng=np_rng,
            parent=parent,
        )
        self.init_param(node_idx, eval_result)

    def hashable(self,):
        return (
            self.__class__.__name__, self.init_transit_code,
            tuple(sorted(self.experiences, key=str,)),
            self.crt_exp_num_per_refine,
        )
    def __str__(self,):
        return f'{self.__class__.__name__}(init code of exp success ratio {self.eval_result["exp_result"]["success_ratio"]})'

    def __call__(self,):
        output = self.main_call()
        log.debug(f'{self.__class__.__name__} called {self.called_times} times')
        self.update_param(output['success_flag'])
        return output
    def main_call(self,):
        exp_result = self.eval_result['exp_result']
        crt_experiences = exp_result['crt_experiences']
        wrong_experiences = exp_result['wrong_experiences']
        assert len(crt_experiences) > 0, f'len(crt_experiences) == 0'
        assert len(wrong_experiences) > 0, f'len(wrong_experiences) == 0'
        if len(crt_experiences) > self.crt_exp_num_per_refine:
            crt_key_indexes = self.local_np_rng.choice(len(crt_experiences), size=self.crt_exp_num_per_refine, replace=False)
            crt_experiences = [crt_experiences[i] for i in crt_key_indexes]
        else:
            crt_experiences = crt_experiences
        wrong_key_indexes = self.local_np_rng.choice(len(wrong_experiences), size=1, replace=False)
        wrong_experiences = [wrong_experiences[i] for i in wrong_key_indexes]
        output = refine_transit(
            code=self.init_transit_code,
            crt_experiences=crt_experiences,
            wrong_experiences=wrong_experiences,
            llm=self.llm,
            verbose=self.verbose,
        )
        output.update(evaluate(output['code'], self.experiences,))
        output['transit_code'] = output['code']
        return output

    def init_param(self, node_idx, eval_result):
        self.called_times = 0
        self.alpha = 1 + eval_result['success_ratio'] * self.bandits_C
        self.beta = 1 + (1 - eval_result['success_ratio']) * self.bandits_C

def synthesize_transit(
    experiences,
    init_transit_code=None,
    llm=None,
    verbose=False,
    max_budget=100, #$100
    bandits_C=25.0,
    np_rng=None,
    with_total_cost=False,
):
    if np_rng is None:
        np_rng = np.random.default_rng(seed=0)
    elif isinstance(np_rng, int):
        np_rng = np.random.default_rng(seed=np_rng)
    assert isinstance(np_rng, np.random.Generator)
    log.debug('~' * 20 + os.path.basename(__file__).replace('.py', '') + '~' * 20)
    total_cost = 0
    new_cost = 0

    actions = [InitAction(
        node_idx=0,
        experiences=experiences,
        llm=llm,
        np_rng=np_rng,
        verbose=verbose,
    )]
    if init_transit_code is not None:
        eval_result = evaluate(
            init_transit_code,
            experiences,
        )
        if eval_result['success_flag']:
            log.debug('Found perfect solution in the beginning')
            return eval_result
        best_success_ratio = eval_result['exp_result']['success_ratio']
        best_outputs = [eval_result,]

        actions.append(RefineTransitAction(
            len(actions),
            init_transit_code,
            experiences,
            llm,
            eval_result=eval_result,
            bandits_C=bandits_C,
            np_rng=np_rng,
            verbose=verbose,
            parent='FAKE',
        ))

    visited_actions = set(actions)

    step = 0
    best_success_ratio = -1
    while total_cost < max_budget:
        log.info('='*10 + f'step {step}' + '='*10)
        
        if step > 0 and step % 100 == 0:
            log.info(f'After {step+1} steps, best success ratio is {best_success_ratio} with {len(best_outputs)} possible solutions. Randomly choose one.')
            output = np_rng.choice(best_outputs)
            log.info(f'Best Code:\n{output["code"]}')

        sampled_theta_list = [action.sample() for action in actions]
        action_index = np.argmax(sampled_theta_list)
        action = actions[action_index]

        log.debug('='*10 + str(action) + '='*10)
        output = action()
        total_cost += output['final_outputs']['costs']['cost']
        new_cost += output['final_outputs']['new_costs']['cost']
        log.info(f'total_cost: {total_cost}, new_cost: {new_cost}')

        exp_result = output['exp_result']
        log.info('exp_result:' + str({k: v for k, v in exp_result.items() if itemnum(v) < 50 and len(str(v)) < 100}))

        if output['exp_result']['success_ratio'] > best_success_ratio:
            best_success_ratio = output['exp_result']['success_ratio']
            best_outputs = [output,]
        elif output['exp_result']['success_ratio'] == best_success_ratio:
            best_outputs.append(output)

        eval_result = output
        if not eval_result['exp_result']['success_flag'] and len(eval_result['exp_result']['crt_experiences']) > 0:
            new_action = RefineTransitAction(
                len(actions),
                output['transit_code'],
                experiences,
                llm,
                eval_result=eval_result,
                bandits_C=bandits_C,
                np_rng=np_rng,
                verbose=verbose,
                parent=action,
            )
            assert new_action == new_action, 'RefineAction must be hashable'
            if new_action not in visited_actions:
                actions.append(new_action)
                visited_actions.add(new_action)
                log.debug(f'Add new action {new_action}')

        if output['success_flag']:
            return output

        step += 1

    log.info(f'After {step+1} steps, best success ratio is {best_success_ratio} with {len(best_outputs)} possible solutions. Randomly choose one.')
    output = np_rng.choice(best_outputs)
    if with_total_cost:
        output['final_total_cost'] = total_cost
    return output
