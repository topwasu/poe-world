import os
import hydra
import logging
import copy
import numpy as np
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Sequence, List, Dict, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod

from classes.helper import Obj, ObjList, SeqValues, RandomValues, ObjSelector, StateTransitionTriplet, StateMemory
from classes.helper import (add_noise_to_obj_list_dist,
                            fill_unset_values_with_uniform,
                            evaluate_logprobs_of_obj_list,
                            combine_obj_list_dists, instantiate_obj_list,
                            replace_objs_w_specified_types,
                            match_two_obj_lists)
from classes.envs.object_tracker import ObjectTracker
from learners.utils import *

log = logging.getLogger('main')


class Model(ABC):
    """Abstract base class for predictive models"""
    @abstractmethod
    def sample_next_scene(self, obj_list_prev: ObjList, event: Any, **kwargs) -> ObjList:
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        pass


def entropy_regularization(params):
    probs = F.softmax(params, dim=0)  # Convert to probability distribution
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))  # Compute entropy
    return entropy


def set_obj2_randomvalues_w_obj1_seqvalues(obj1, idx, obj2) -> None:
    """
    Set obj2 attribute to a RandomValue whose value is equal to the 
    idx-th value of obj1's attribute which is a SeqValue
    Return whether at least one attribute gets set
    """
    success = True
    if isinstance(obj1.velocity_x, SeqValues):
        obj2.velocity_x = RandomValues([obj1.velocity_x.sequence[idx]])
    elif isinstance(obj1.velocity_y, SeqValues):
        obj2.velocity_y = RandomValues([obj1.velocity_y.sequence[idx]])
    elif isinstance(obj1.deleted, SeqValues):
        obj2.deleted = RandomValues([obj1.deleted.sequence[idx]])
    elif isinstance(obj1.w_change, SeqValues):
        obj2.w_change = RandomValues([obj1.w_change.sequence[idx]])
    elif isinstance(obj1.h_change, SeqValues):
        obj2.h_change = RandomValues([obj1.h_change.sequence[idx]])
    else:
        success = False
    return success


class MoEObjModel(Model):
    """
    Mixture of Experts Object Model
    Combines multiple expert models to predict object states
    Have two modes: non-creation (dealing with existing objects) and creation (dealing with new objects)
    """
    def __init__(self,
                 name: str,
                 config: Any,
                 rules: List[str] = [],
                 obj_type: Optional[str] = None,
                 objects_selector: Optional[ObjSelector] = None,
                 size_change_flag: bool = False):
        """
        Initialize MoE model
        
        Args:
            name: Model name
            config: Configuration object
            rules: List of rule strings
            program_name: Name of program to execute
            objects_selector: Function to select relevant objects
        """
        self.name = name
        self.config = config
        self.rules = list(dict.fromkeys(rules))
        self.params = [0.5] * len(self.rules)
        self.context_lengths = [-1] * len(self.rules)
        self.fitteds = [False] * len(rules)
        self.obj_type = obj_type
        self.program_name = f'alter_{self.obj_type}_objects'
        self.objects_selector = objects_selector
        self.callables = []
        self.precompute_dist = []

        # Set size change flag
        self.size_change_flag = size_change_flag
        self.init_helper_funcs_with_size_change_flag()

        self.cache_enabled = False
        self.cache = {}

        self._prep_callables()

    def _remove_id_obj_type_color_change(self, rule: str) -> str:
        new_rule = '\n'.join([
            x if '.id = ' not in x and '.obj_type = ' not in x
            and '.color = ' not in x else x[:-len(x.lstrip(' '))] + 'pass'
            for x in rule.split('\n')
        ])

        if new_rule.startswith(f'def {self.program_name}'):
            new_rule = new_rule[:len(f'def {self.program_name}')] + new_rule[
                new_rule.find('(obj_list: ObjList,'):]
        else:
            obj_type = new_rule[8:].split('_', 1)[0]
            new_rule = new_rule[:len(f'def get_{obj_type}_objects'
                                     )] + new_rule[new_rule.
                                                   find('(obj_list: ObjList'):]
        return new_rule
        # return '\n'.join(filter(lambda x: '.id = ' not in x and '.obj_type = ' not in x and '.color = ' not in x, rule.split('\n')))

    def _add_touch_params(self, rule: str) -> str:
        params_set = True if 'touch_side' in rule.split('\n')[0] else False
        if not params_set:
            new_rule = ''
            for x in rule.split('\n'):
                if 'obj_list: ObjList' in x:
                    idx = x.find(')')
                    x = x[:idx] + ', touch_side=-1, touch_percent=0.1' + x[idx:]
                elif '.touches(' in x:
                    idx1 = x.find('.touches(')
                    idx2 = x[idx1:].find(')')
                    idx = idx1 + idx2
                    x = x[:idx] + ', touch_side, touch_percent' + x[idx:]
                new_rule += x + '\n'
            return new_rule
        else:
            return rule

    def _prep_callable_single_rule(self, rule: str) -> Optional[Any]:
        all_context_vars = {**globals(), **locals()}
        try:
            exec(rule, all_context_vars)
        except:
            if '\nAND\n' in rule:
                try:
                    condition_txt, effect_txt = rule.split('\nAND\n')
                    exec(condition_txt, all_context_vars)
                    exec(effect_txt, all_context_vars)
                    obj_type = condition_txt[8:].split('_', 1)[0]
                    condition = all_context_vars[f'get_{obj_type}_objects']
                    effect = all_context_vars[self.program_name]
                    # Assume this weird, restart rule does not contain size change
                    return (lambda x, _, *args: effect(x, None)
                            if condition(x, *args) else
                            add_noise_to_obj_list_dist(x, make_uniform=True))
                except:
                    log.warning('SOMETHING REALLY WRONG')
                    log.warning(traceback.format_exc())
                    return None
            else:
                log.warning(rule)
                log.warning('SOMETHING REALLY WRONG')
                log.warning(traceback.format_exc())
                return None

        try:
            return all_context_vars[self.program_name]
        except:
            log.warning(
                f'Cannot find {self.program_name} program after executing llm-generated code, probably bad func name'
            )
            return None

    def _prep_callables(self) -> None:
        while len(self.callables) != len(self.rules):
            rule = self.rules[len(self.callables)]
            rule = self._remove_id_obj_type_color_change(rule)
            rule = self._add_touch_params(rule)
            self.rules[len(
                self.callables
            )] = rule  # Replace old rule with newly processed rule

            rule_callable = self._prep_callable_single_rule(rule)

            if rule_callable is not None:
                self.callables.append(rule_callable)
            else:
                del self.rules[len(self.callables)]
                del self.params[len(self.callables)]
                del self.context_lengths[len(self.callables)]
                del self.fitteds[len(self.callables)]

    # IMPORTANT only use use_precompute if can and needed to
    def _objective(self,
                   params: Union[List[float], torch.Tensor],
                   c: List[Any],
                   use_torch: bool = False,
                   indices: Optional[List[int]] = None,
                   use_precompute: bool = False) -> float:
        if use_torch:
            params = torch.clamp(params, min=0, max=10)
        else:
            params = np.clip(params, 0, 10)
        if indices is None:
            indices = range(len(c))

        if use_precompute is False:
            raise Exception(
                'Objective now only supports use_precompute being True.'
                'Use evaluate logprobs instead for use_precompute being False')
            # TODO: eventually remove use_precompute arguments

        res = 0
        for idx in indices:
            x = c[idx]

            if len(self.precompute_dist) <= idx:
                raise Exception('idx given is not in precompute_dist')

            value = self.evaluate_logprobs(
                x.input_state,
                x.event,
                x.output_state,
                params=params,
                use_torch=use_torch,
                precompute_index=idx
                if len(self.precompute_dist) > idx and use_precompute else -1)

            res += value
        return -res / len(c) * 1000

    def _objective_individual(self, params: List[float], c: List[Any],
                              callable: Any, context_length: int) -> float:
        # TODO Double check
        if context_length == -1:
            x = c[-1]
            objs = self.objects_selector(x.output_state, x.input_state)

            pre_stepped_obj_list_prev = x.input_state.deepcopy()
            pre_stepped_obj_list_prev.pre_step()
            obj_list_dist = callable(pre_stepped_obj_list_prev, x.event,
                                     *params)
            obj_list_dist = self.objects_selector(obj_list_dist,
                                                  pre_stepped_obj_list_prev)
            self.fill_unset_values_with_uniform(obj_list_dist)
            self.add_noise_to_obj_list_dist(obj_list_dist)
            obj_list_dist.step()

            sm_logprobs = self.evaluate_logprobs_of_obj_list(
                obj_list_dist, objs, by_pos=(self.name == 'creation'))
        else:
            sm_logprobs = 0

            context_c = c[-context_length:]
            objs_list = [
                self.objects_selector(x.output_state, x.input_state)
                for x in context_c
            ]

            pre_stepped_obj_list_prev = context_c[0].input_state.deepcopy()
            pre_stepped_obj_list_prev.pre_step()
            obj_list_seqval = callable(pre_stepped_obj_list_prev.deepcopy(),
                                       context_c[0].event, *params)
            obj_list_seqval.step()  #?

            for idx in range(context_length):
                obj_list_dist = context_c[idx].input_state.deepcopy()
                obj_list_dist.pre_step()
                for obj in obj_list_seqval:  # Speed it up
                    try:
                        matched_obj = obj_list_dist.get_obj_by_id(obj.id)
                    except:
                        continue

                    set_obj2_randomvalues_w_obj1_seqvalues(
                        obj, idx, matched_obj)

                # obj_list_dist = self.objects_selector(obj_list_dist, pre_stepped_obj_list_prev)
                obj_list_dist = self.objects_selector(
                    obj_list_dist, context_c[idx].input_state)
                self.fill_unset_values_with_uniform(obj_list_dist)
                self.add_noise_to_obj_list_dist(obj_list_dist)
                obj_list_dist.step()

                # log.info(f'Pre: {pre_stepped_obj_list_prev.get_player_interactions()}')
                # log.info(f'Predicted Seq: {obj_list_seqval[0].velocity_y}')
                # log.info(f'Predicted: {obj_list_dist[0].velocity_y}')
                # log.info(f'Actual: {objs_list[idx][0].velocity_y}')
                sm_logprobs += self.evaluate_logprobs_of_obj_list(
                    obj_list_dist,
                    objs_list[idx],
                    by_pos=(self.name == 'creation'))

        return -sm_logprobs / len(c) * 1000

    def extend_rules(self,
                     rules: List[str],
                     c: List[Any],
                     context_lengths: List[int] = [],
                     optimize_touch_params=True) -> None:
        """
        Extend model with new rules
        
        Args:
            rules: New rules to add
            c: Training examples to fit parameters
            context_lengths: context lengths of the rules
        """
        if len(context_lengths) == 0:
            context_lengths = [-1] * len(rules)

        # Always optimize param
        log.debug(f'Extending {len(rules)} rules')

        for rule, context_length in zip(rules, context_lengths):
            if optimize_touch_params:
                try:
                    rule = self._remove_id_obj_type_color_change(rule)
                    rule = self._add_touch_params(rule)
                    callable = self._prep_callable_single_rule(rule)
                    
                    if 'if not ' not in rule and ' and not ' not in rule: # If this is not a not 'touch anything' rule
                        best_param = (0, 0.1)  # Has to pick side

                        mn = self._objective_individual([0, 0.1], c, callable,
                                                        context_length)
                        for touch_side in range(4):
                            for touch_percent in np.arange(0.1, 1.1, 0.1):
                                obj = self._objective_individual(
                                    [touch_side, touch_percent], c, callable,
                                    context_length)
                                if obj <= mn:
                                    if obj < mn:
                                        best_param = (touch_side, touch_percent)
                                    elif touch_percent > best_param[1]:
                                        best_param = (touch_side, touch_percent)
                                    mn = obj
                        rule = self._set_param_value(rule, 'touch_side', best_param[0])
                        rule = self._set_param_value(rule, 'touch_percent',
                                                    best_param[1])
                    if rule not in self.rules:
                        self.rules.append(rule)
                        self.context_lengths.append(context_length)
                except:
                    log.warning(f'Error while trying to execute a rule')
                    log.warning(traceback.format_exc())
            else:
                if rule not in self.rules:
                    self.rules.append(rule)
                    self.context_lengths.append(context_length)

        log.debug(f'Done extending {len(rules)} rules')
        n_new_rules = (len(self.rules) - len(self.params))
        self.params = self.params + ([0.5] * n_new_rules)
        self.fitteds = self.fitteds + ([False] * n_new_rules)
        self._prep_callables()  # prep callable again to update rule

    def init_helper_funcs_with_size_change_flag(self) -> None:
        """
        Update self.size_change_flag to indicate whether we have a size change rule
        This can be optimized
        """
        self.add_noise_to_obj_list_dist = lambda *args, **kwargs: add_noise_to_obj_list_dist(
            *args, **kwargs, size_change_flag=self.size_change_flag)
        self.fill_unset_values_with_uniform = lambda *args, **kwargs: fill_unset_values_with_uniform(
            *args, **kwargs, size_change_flag=self.size_change_flag)
        self.evaluate_logprobs_of_obj_list = lambda *args, **kwargs: evaluate_logprobs_of_obj_list(
            *args, **kwargs, size_change_flag=self.size_change_flag)
        self.combine_obj_list_dists = lambda *args, **kwargs: combine_obj_list_dists(
            *args, **kwargs, size_change_flag=self.size_change_flag)

    def _get_param_value(self, rule: str, param_name: str) -> float:
        idx1 = rule.find(f'{param_name}=')
        idx2 = min(rule[idx1:].find(','), rule[idx1:].find(')'))
        if idx2 == -1:
            idx2 = max(rule[idx1:].find(','), rule[idx1:].find(')'))

        return float(rule[idx1 + len(f'{param_name}='):idx1 + idx2])

    def _set_param_value(self, rule: str, param_name: str,
                         value: float) -> str:
        idx1 = rule.find(f'{param_name}=')
        idx2 = min(rule[idx1:].find(','), rule[idx1:].find(')'))
        if idx2 == -1:
            idx2 = max(rule[idx1:].find(','), rule[idx1:].find(')'))
        return rule[:idx1 + len(f'{param_name}=')] + f'{value}' + rule[idx1 +
                                                                       idx2:]

    def prune_programs_with_c(self, c: List[Any]) -> None:
        if len(self.rules) == 0:
            log.info('No rules to prune')
            return
        
        log.info('Prunning no contribution programs')
        new_rules = []
        new_callables = []
        new_params = []
        new_fitteds = []
        new_context_lengths = []
        new_indices = []
        current_obj = self._objective(self.params, c, use_precompute=True)
        for idx, (param, rule, callable, fitted, context_length) in enumerate(
                zip(self.params, self.rules, self.callables, self.fitteds,
                    self.context_lengths)):
            new_obj = self._objective(self.params[:idx] + [0] +
                                      self.params[idx + 1:],
                                      c,
                                      use_precompute=True)
            if new_obj <= current_obj + 1:
                current_obj = new_obj
                self.params[idx] = 0
            else:
                new_params.append(param)
                new_rules.append(rule)
                new_callables.append(callable)
                new_fitteds.append(fitted)
                new_context_lengths.append(context_length)
                new_indices.append(idx)
        if len(new_params) == 0:
            new_params.append(self.params[-1])
            new_rules.append(self.rules[-1])
            new_callables.append(self.callables[-1])
            new_fitteds.append(self.fitteds[-1])
            new_context_lengths.append(self.context_lengths[-1])
            new_indices.append(-1)
        log.info(
            f'Done pruning no contribution programs (n = {len(self.params)} -> {len(new_params)})'
        )
        self.params = new_params
        self.rules = new_rules
        self.callables = new_callables
        self.context_lengths = new_context_lengths
        self.fitteds = new_fitteds

        for idx, objs_dists in enumerate(self.precompute_dist):
            self.precompute_dist[idx] = [
                objs_dists[idx] for idx in new_indices
            ]

    def prune_programs(self) -> None:
        log.info('Prunning zero / negative weight programs')
        new_rules = []
        new_callables = []
        new_params = []
        new_fitteds = []
        new_context_lengths = []
        new_indices = []
        for idx, (param, rule, callable, fitted, context_length) in enumerate(
                zip(self.params, self.rules, self.callables, self.fitteds,
                    self.context_lengths)):
            if param > 1e-2:
                new_params.append(param)
                new_rules.append(rule)
                new_callables.append(callable)
                new_fitteds.append(fitted)
                new_context_lengths.append(context_length)
                new_indices.append(idx)
        if len(new_params) == 0:
            new_params.append(self.params[-1])
            new_rules.append(self.rules[-1])
            new_callables.append(self.callables[-1])
            new_context_lengths.append(self.context_lengths[-1])
            new_fitteds.append(self.fitteds[-1])
            new_indices.append(-1)
        log.info(
            f'Done pruning zero / negative weight programs (n = {len(self.params)} -> {len(new_params)})'
        )
        self.params = new_params
        self.rules = new_rules
        self.callables = new_callables
        self.context_lengths = new_context_lengths
        self.fitteds = new_fitteds

        for idx, objs_dists in enumerate(self.precompute_dist):
            self.precompute_dist[idx] = [
                objs_dists[idx] for idx in new_indices
            ]

    def fit_weights(self, c: List[StateTransitionTriplet], include_l1_loss: bool = True) -> None:
        """
        Fit mixture weights using training examples
        
        Args:
            c: Training examples
        """
        if len(self.rules) == 0:
            log.info('No rules to fit')
            return
        
        log.debug('Before fitting weights -- pruning bad programs...')
        before_len = len(self.rules)
        for x in c:
            self._prune_bad_rules(x.input_state, x.event)
        after_len = len(self.rules)
        log.debug(f'Pruned {after_len - before_len} bad programs')

        log.info('Precompute unweighted distributions...')
        # ASSUMPTION: precompute_dist[idx] (if available) corresponds to c[idx]
        log.info(
            f'length precompute {len(self.precompute_dist)} length c {len(c)}')
        for idx, x in enumerate(c):
            # haven't seen this example
            memory = x.input_state.memory
            
            if len(self.precompute_dist) <= idx:
                self.precompute_dist.append(
                    self._get_obj_list_dists(x.input_state,
                                             x.event,
                                             memory))

            # old rules have seen this example but new rules have not
            if len(self.precompute_dist[idx]) < len(self.callables):
                pre_stepped_obj_list_prev = x.input_state.deepcopy()
                pre_stepped_obj_list_prev.pre_step()
                while len(self.precompute_dist[idx]) < len(self.callables):
                    self.precompute_dist[idx].append(
                        self._get_obj_list_dists_helper(
                            pre_stepped_obj_list_prev, x.event,
                            self.callables[len(self.precompute_dist[idx])],
                            self.context_lengths[len(self.precompute_dist[idx])],
                            memory))
        log.info(f'done length precompute {len(self.precompute_dist)}')

        new_params = list(self._fit_weights_helper(c, include_l1_loss=include_l1_loss))
        self.params = new_params
        self.fitteds = [True] * len(self.params)

    def fit_only_new_weights(self, c: List[Any], include_l1_loss: bool = True) -> None:
        if len(self.rules) == 0:
            log.info('No rules to fit')
            return
        
        log.info('Precompute unweighted distributions...')
        # ASSUMPTION: precompute_dist[idx] (if available) corresponds to c[idx]
        log.info(
            f'length precompute {len(self.precompute_dist)} length c {len(c)}')
        for idx, x in enumerate(c):
            memory = x.input_state.memory
            # haven't seen this example
            if len(self.precompute_dist) <= idx:
                self.precompute_dist.append(
                    self._get_obj_list_dists(x.input_state,
                                             x.event,
                                             memory))

            # old rules have seen this example but new rules have not
            if len(self.precompute_dist[idx]) < len(self.callables):
                pre_stepped_obj_list_prev = x.input_state.deepcopy()
                pre_stepped_obj_list_prev.pre_step()
                while len(self.precompute_dist[idx]) < len(self.callables):
                    self.precompute_dist[idx].append(
                        self._get_obj_list_dists_helper(
                            pre_stepped_obj_list_prev, x.event,
                            self.callables[len(self.precompute_dist[idx])],
                            self.context_lengths[len(self.precompute_dist[idx])],
                            memory))
        log.info(f'done length precompute {len(self.precompute_dist)}')

        freeze_before = -1
        for idx in range(len(self.fitteds) - 1, -1, -1):
            if self.fitteds[idx]:
                freeze_before = idx
                break
            self.params[idx] = 0.01
        new_params = list(
            self._fit_weights_helper(c, freeze_before=freeze_before, include_l1_loss=include_l1_loss))
        self.params = new_params
        self.fitteds = [True] * len(self.params)

    def _fit_weights_helper(self,
                            c: List[Any],
                            freeze_before: int = -1,
                            include_l1_loss: bool = True) -> List[float]:
        log.info('Fitting weights...')
        log.debug(f'Number of rules = {len(self.rules)}')
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        if freeze_before != -1:
            weights = nn.Parameter(
                torch.tensor([1.0] + self.params[freeze_before + 1:],
                             dtype=torch.float32,
                             device=device))
            old_precompute_dist = []
            for idx, x in enumerate(c):
                old_precompute_dist.append(self.precompute_dist[idx].copy())
                memory = x.input_state.memory
                self.precompute_dist[idx] = [
                    self.get_next_scene_distributions(
                        x.input_state,
                        x.event,
                        memory,
                        params=self.params[:freeze_before + 1] + [0.0] *
                        (len(weights) - freeze_before - 1),
                        precompute_index=idx)
                ] + self.precompute_dist[idx][freeze_before + 1:]
            mask = torch.tensor([0.0] + [1.0] *
                                (len(self.params) - freeze_before - 1),
                                dtype=torch.float32,
                                device=device)
        else:
            if self.config.moe.continue_params:
                weights = nn.Parameter(
                    torch.tensor(self.params,
                                 dtype=torch.float32,
                                 device=device))
            else:
                weights = nn.Parameter(
                    torch.tensor(np.ones_like(self.params) * 0.5,
                                 dtype=torch.float32,
                                 device=device))
            mask = torch.tensor([1.0] * len(weights),
                                dtype=torch.float32,
                                device=device)
        if self.config.moe.optim == 'adam':
            optimizer = optim.Adam([weights], lr=self.config.moe.lr)
        elif self.config.moe.optim == 'lbfgs':
            optimizer = optim.LBFGS([weights],
                                    lr=self.config.moe.lr,
                                    line_search_fn='strong_wolfe')
        else:
            raise NotImplementedError
        
        l1_weight = 1 if include_l1_loss else 0

        n_steps = self.config.moe.n_steps
        # n_steps = self.config.moe.n_steps if freeze_before == -1 else max(
        #     1, self.config.moe.n_steps // 10)
        batch_size = self.config.moe.batch_size
        for step in range(n_steps):
            log.debug(f'Epoch {step}')
            log.debug(len(c))

            # random.shuffle(c)
            # for i in range(0, len(c), 32):
            def closure():
                optimizer.zero_grad(
                )  # Zero the gradients from previous iteration
                with torch.set_grad_enabled(True):
                    # Compute the objective function (loss)
                    loss = self._objective(weights,
                                           c,
                                           use_torch=True,
                                           indices=np.random.choice(len(c), batch_size, replace=False) if batch_size < len(c) else np.arange(len(c)),
                                           use_precompute=True) \
                                               + l1_weight * weights.abs().sum() # might need to deal with redundant rules

                    # Compute gradients
                    loss.backward()

                    # Mask out the things we want to freeze
                    weights.grad *= mask
                return loss

            optimizer.step(closure)
            if step % 50 == 0 or self.config.moe.optim == 'lbfgs':
                log.info(f'Step {step} Loss {closure().item()}')
        log.info(f'Done\nOld weights {self.params}\nNew weights {weights}')
        log.info(
            f'Final objective = {self._objective(weights, c, use_torch=True, use_precompute=True)}'
        )

        if freeze_before != -1:
            self.precompute_dist = old_precompute_dist
            return self.params[:freeze_before + 1] + list(
                np.asarray(weights.detach().cpu().numpy()))[1:]
        else:
            return list(np.asarray(weights.detach().cpu().numpy()))

    def _get_obj_list_dists(
            self,
            obj_list_prev: ObjList,
            event: Any,
            memory: StateMemory) -> List[ObjList]:
        if self.cache_enabled:
            k = f'{obj_list_prev}{event}'
            for i in range(min(self.config.moe.cache_history_size, len(memory))):
                k += f'{memory[-i-1][0]}{memory[-i-1][1]}'
            if k in self.cache:
                return self.cache[k]

        objs_dists = []

        pre_stepped_obj_list_prev = obj_list_prev.deepcopy()
        pre_stepped_obj_list_prev.pre_step()

        for idx, (callable, rule, context_length) in enumerate(
                zip(self.callables, self.rules, self.context_lengths)):
            objs_dists.append(
                self._get_obj_list_dists_helper(pre_stepped_obj_list_prev,
                                                event, callable,
                                                context_length, memory))

        if self.cache_enabled:
            k = f'{obj_list_prev}{event}'
            for i in range(min(self.config.moe.cache_history_size, len(memory))):
                k += f'{memory[-i-1][0]}{memory[-i-1][1]}'
            self.cache[k] = objs_dists

        return objs_dists

    def _get_obj_list_dists_helper(self, pre_stepped_obj_list_prev: ObjList,
                                   event: Any, callable: Any,
                                   context_length: int,
                                   memory: StateMemory) -> Optional[ObjList]:
        try:
            if context_length == -1:  # fully observable MDP program
                obj_list_next_dist = callable(
                    pre_stepped_obj_list_prev.deepcopy(), event)
                objs_dist = self.objects_selector(obj_list_next_dist,
                                                  pre_stepped_obj_list_prev)

                found_one_set = self.fill_unset_values_with_uniform(objs_dist)
                if not found_one_set:
                    return None
                self.add_noise_to_obj_list_dist(objs_dist)
                objs_dist.step()

                return objs_dist
            else:  # pomdp program
                for idx in range(context_length):
                    if idx == 0:
                        input_state = pre_stepped_obj_list_prev
                        input_event = event
                    else:
                        input_state = memory[-idx][0]
                        input_event = memory[-idx][1]
                    obj_list_seqval = callable(input_state.deepcopy(),
                                               input_event)
                    obj_list_next_dist = pre_stepped_obj_list_prev.deepcopy()

                    found = False
                    new_objs = []
                    for obj in obj_list_seqval:
                        try:
                            matched_obj = obj_list_next_dist.get_obj_by_id(
                                obj.id)

                            res = set_obj2_randomvalues_w_obj1_seqvalues(
                                obj, idx, matched_obj)
                            if res:
                                found = True
                        except:  # No match
                            # If not match -- maybe it's a new object so check if it's a new object
                            if isinstance(obj.deleted, SeqValues
                                          ) and obj.deleted.sequence[idx] == 0:
                                obj.deleted = RandomValues([0])
                                new_objs.append(obj)
                                found = True

                    if len(new_objs) > 0:
                        obj_list_next_dist = ObjList(new_objs, no_copy=True)

                    if found:
                        break

                objs_dist = self.objects_selector(obj_list_next_dist,
                                                  pre_stepped_obj_list_prev)

                found_one_set = self.fill_unset_values_with_uniform(objs_dist)
                if not found_one_set:
                    return None
                self.add_noise_to_obj_list_dist(objs_dist)
                objs_dist.step()

                return objs_dist
        except:
            return None

    def _prune_bad_rules(self, obj_list_prev: ObjList, event: Any) -> None:
        pre_stepped_obj_list_prev = obj_list_prev.deepcopy()
        pre_stepped_obj_list_prev.pre_step()

        bad_indices = []
        for idx, callable in enumerate(self.callables):
            try:
                callable(pre_stepped_obj_list_prev.deepcopy(), event)
            except:
                log.warning(f'Error while trying to execute a rule')
                log.warning(traceback.format_exc())
                bad_indices.append(idx)

        self.callables = [
            self.callables[idx] for idx in range(len(self.callables))
            if idx not in bad_indices
        ]
        self.rules = [
            self.rules[idx] for idx in range(len(self.rules))
            if idx not in bad_indices
        ]
        self.params = [
            self.params[idx] for idx in range(len(self.params))
            if idx not in bad_indices
        ]
        self.context_lengths = [
            self.context_lengths[idx]
            for idx in range(len(self.context_lengths))
            if idx not in bad_indices
        ]
        self.fitteds = [
            self.fitteds[idx] for idx in range(len(self.fitteds))
            if idx not in bad_indices
        ]

    def get_next_scene_distributions(self,
                                     obj_list_prev: ObjList,
                                     event: Any,
                                     memory: Optional[StateMemory] = None,
                                     params: Optional[Union[
                                         List[float], torch.Tensor]] = None,
                                     use_torch: bool = False,
                                     precompute_index: int = -1) -> ObjList:
        if params is None:
            if self.params is None:
                raise Exception(
                    "self.params can't be None when params is None")
            params = self.params
            
        if use_torch:
            params = torch.clamp(params, min=0, max=10)
        else:
            params = np.clip(params, 0, 10)

        if self.name == 'non_creation' and len(
                self.objects_selector(obj_list_prev)) == 0:
            return ObjList([])

        if precompute_index == -1:
            if memory is None:
                raise Exception('Memory is required for non-precompute')
            objs_dists = self._get_obj_list_dists(obj_list_prev,
                                                  event,
                                                  memory)
        else:
            objs_dists = self.precompute_dist[precompute_index]

        if use_torch:
            indices = [objs_dist is not None for objs_dist in objs_dists]
            good_params = params[indices]
        else:
            good_params = [
                param for objs_dist, param in zip(objs_dists, params)
                if objs_dist is not None
            ]
        objs_dists = [
            objs_dist for objs_dist in objs_dists if objs_dist is not None
        ]

        if self.name == 'creation':
            new_objs_dists = []
            for objs_dist in objs_dists:
                new_objs_dist = objs_dist
                if len(objs_dist) == 0:
                    new_objs_dist = new_objs_dist.create_object(
                        self.obj_type, 0, 0)
                    new_objs_dist[0].deleted = RandomValues([1])
                    self.fill_unset_values_with_uniform(new_objs_dist)
                    self.add_noise_to_obj_list_dist(new_objs_dist)
                new_objs_dists.append(new_objs_dist)
            # Replace the old objs_dists with the new ones
            objs_dists = new_objs_dists

        final_objs_dist = self.combine_obj_list_dists(
            objs_dists,
            good_params,
            use_torch,
            padding=(self.name == 'creation'))

        return final_objs_dist

    def sample_next_scene(
            self,
            obj_list_prev: ObjList,
            event: Any,
            memory: StateMemory,
            params: Optional[Union[List[float],
                                   torch.Tensor]] = None) -> ObjList:
        obj_list_dist = self.get_next_scene_distributions(obj_list_prev,
                                                          event,
                                                          memory,
                                                          params=params)
        return instantiate_obj_list(obj_list_dist)

    def evaluate_logprobs(self,
                          obj_list_prev: ObjList,
                          event: Any,
                          obj_list_next: ObjList,
                          memory: Optional[StateMemory] = None,
                          params: Optional[Union[List[float],
                                                 torch.Tensor]] = None,
                          use_torch: bool = False,
                          precompute_index: int = -1) -> float:
        objs = self.objects_selector(obj_list_next, obj_list_prev)
        obj_list_dist = self.get_next_scene_distributions(
            obj_list_prev,
            event,
            memory=memory,
            params=params,
            use_torch=use_torch,
            precompute_index=precompute_index)
        res = self.evaluate_logprobs_of_obj_list(
            obj_list_dist, objs, by_pos=(self.name == 'creation'))
        return res

    def save(self, file_path: str) -> None:
        raise NotImplementedError

    def load(self, file_path: str) -> None:
        raise NotImplementedError

    def clear_cache(self) -> None:
        self.cache = {}

    def enable_cache(self) -> None:
        self.clear_cache()
        self.cache_enabled = True

    def disable_cache(self) -> None:
        self.clear_cache()
        self.cache_enabled = False

    def clear_precompute_dist(self) -> None:
        self.precompute_dist = []

    def remove_callables(self) -> 'MoEObjModel':
        new_instance = MoEObjModel(self.name,
                                   self.config,
                                   obj_type=self.obj_type,
                                   objects_selector=self.objects_selector)
        new_instance.rules = self.rules
        new_instance.params = self.params
        new_instance.context_lengths = self.context_lengths
        new_instance.fitteds = self.fitteds
        new_instance.precompute_dist = self.precompute_dist
        return new_instance

    def prepare_callables(self) -> None:
        self._prep_callables()

    def __len__(self):
        return len(self.rules)


class Constraints:
    """Class to handle object interaction constraints"""
    def __init__(self, obj_type: str, interactions_selector: Any):
        """
        Initialize constraints
        
        Args:
            obj_type: Type of object to constrain
            interactions_selector: Function to select valid interactions
        """
        self.obj_type = obj_type
        self.interactions_selector = interactions_selector
        self.rules = []
        self.callables = []
        self.small_cache = {}

    def _objective_individual(self, params: List[float], x: Any,
                              callable: Any) -> float:
        obj_list = x.output_state.deepcopy()
        touch_ids, satisfied_ids = callable(obj_list, None, *params)
        if len(touch_ids) > 0 and len(touch_ids) == len(satisfied_ids):
            return -1
        return 0

    def _objective_sm(self, params: List[float], c: List[Any]) -> float:
        sm = 0
        for x in c:
            no_touch = True
            succeed = False
            for param, rule, callable in zip(params, self.rules,
                                             self.callables):
                if param == 0:
                    continue
                obj_list = x.output_state.deepcopy()
                # k = f'{obj_list}{rule}'
                # if k not in self.small_cache:
                #     self.small_cache[k] = callable(obj_list, None)
                # n_touch, n_satisfied = self.small_cache[k]
                touch_list, satisfied_list = callable(obj_list, None)
                n_touch, n_satisfied = len(touch_list), len(satisfied_list)
                if n_touch:
                    no_touch = False
                    if n_touch == n_satisfied:
                        succeed = True
                        break
            if succeed:
                sm += 1
        return -sm

    def extend_rules(self, rules: List[str], x: Any) -> None:
        log.debug(f'Extending {len(rules)} constraints')
        for rule in rules:
            try:
                rule = self._add_touch_params(rule)
                callable = self._prep_callable_single_rule(rule)
                if callable is None:
                    continue
                best_param = (0, 0.1)
                mn = self._objective_individual([0, 0.1], x, callable)
                for touch_side in range(4):
                    for touch_percent in np.arange(0.1, 1.1, 0.1):
                        obj = self._objective_individual(
                            [touch_side, touch_percent], x, callable)
                        if obj <= mn:
                            if obj < mn:
                                best_param = (touch_side, touch_percent)
                            elif touch_percent > best_param[1]:
                                best_param = (touch_side, touch_percent)
                            mn = obj
                rule = self._set_param_value(rule, 'touch_side', best_param[0])
                rule = self._set_param_value(rule, 'touch_percent',
                                             best_param[1])
                if rule not in self.rules:
                    self.rules.append(rule)
                    self.callables.append(
                        self._prep_callable_single_rule(rule))
            except:
                log.warning(f'Error while trying to execute a rule')
                log.warning(traceback.format_exc())

        log.debug(f'Done extending {len(rules)} constraints')

    def _add_touch_params(self, rule: str) -> str:
        params_set = True if 'touch_side' in rule.split('\n')[0] else False
        if not params_set:
            new_rule = ''
            for x in rule.split('\n'):
                if 'obj_list: ObjList' in x:
                    idx = x.find(')')
                    x = x[:idx] + ', touch_side=-1, touch_percent=0.1' + x[idx:]
                elif '.touches(' in x:
                    idx1 = x.find('.touches(')
                    idx2 = x[idx1:].find(')')
                    idx = idx1 + idx2
                    x = x[:idx] + ', touch_side, touch_percent' + x[idx:]
                new_rule += x + '\n'
            return new_rule
        else:
            return rule

    def _prep_callable_single_rule(self, rule: str) -> Optional[Any]:
        all_context_vars = {**globals(), **locals()}
        try:
            exec(rule, all_context_vars)
            if rule.startswith(f'def check_x_of_{self.obj_type}_objects'):
                return all_context_vars[f'check_x_of_{self.obj_type}_objects']
            elif rule.startswith(f'def check_y_of_{self.obj_type}_objects'):
                return all_context_vars[f'check_y_of_{self.obj_type}_objects']
            else:
                raise NotImplementedError
        except:
            log.warning(
                f'Cannot find program name after executing llm-generated code, probably bad func name'
            )
            log.warning(traceback.format_exc())
            return None

    def _set_param_value(self, rule: str, param_name: str,
                         value: float) -> str:
        idx1 = rule.find(f'{param_name}=')
        idx2 = min(rule[idx1:].find(','), rule[idx1:].find(')'))
        if idx2 == -1:
            idx2 = max(rule[idx1:].find(','), rule[idx1:].find(')'))
        return rule[:idx1 + len(f'{param_name}=')] + f'{value}' + rule[idx1 +
                                                                       idx2:]

    def prune_programs(self, c: List[Any]) -> None:
        log.info('Prunning no effect programs')
        new_rules = []
        new_callables = []
        new_axes = []
        params = [1] * len(self.rules)
        current_obj = self._objective_sm(params, c)
        for idx in range(len(self.rules) - 1, -1, -1):
            new_obj = self._objective_sm(params[:idx] + [0] + params[idx + 1:],
                                         c)
            if new_obj <= current_obj + len(c) * 0.01:
                current_obj = new_obj
                params[idx] = 0
            else:
                new_rules.append(self.rules[idx])
                new_callables.append(self.callables[idx])
        log.info(
            f'Done pruning no effect programs (n = {len(self.rules)} -> {len(new_rules)})'
        )
        self.rules = new_rules
        self.callables = new_callables

    def apply(self,
              obj_list: ObjList,
              target_n_touchs: Optional[List[int]] = None
              ) -> Tuple[bool, List[int]]:
        """
        Apply constraints to object list
        
        Args:
            obj_list: List of objects
            target_n_touchs: Target number of touches for each constraint
            
        Returns:
            success: Whether constraints are satisfied
            n_touchs: Number of touches for each constraint
        """
        n_touchs = []
        succeed = False
        for idx, callable in enumerate(self.callables):
            touch_list, satisfied_list = callable(obj_list, None)
            n_touch, n_satisfied = len(touch_list), len(satisfied_list)
            n_touchs.append(n_touch)
            if target_n_touchs is not None and target_n_touchs[idx] == 0:
                continue
            if n_touch > 0 and n_satisfied == n_touch:
                succeed = True
        return ((sum(n_touchs) == 0 and target_n_touchs is None)
                or succeed), n_touchs

    def get_features(self, obj_list: ObjList) -> List[int]:
        features = []
        for idx, callable in enumerate(self.callables):
            touch_list, satisfied_list = callable(obj_list, None)
            n_touch, n_satisfied = len(touch_list), len(satisfied_list)
            if n_satisfied > 0:
                features.append(satisfied_list[0])
            else:
                features.append(-1)
        return features

    def __len__(self) -> int:
        return len(self.callables)

    def remove_callables(self) -> 'Constraints':
        new_instance = Constraints(self.obj_type, self.interactions_selector)
        new_instance.rules = self.rules
        return new_instance

    def prepare_callables(self) -> None:
        self.callables = [
            self._prep_callable_single_rule(rule) for rule in self.rules
        ]


class ObjTypeModel(Model):
    """
    Full model for an object type, e.g., player, wall, beam, etc.
    Composed of a non-creation model, a creation model, and constraints for that object type
    """
    def __init__(self, obj_type: str, non_creation_model: MoEObjModel,
                 creation_model: MoEObjModel,
                 constraints: Constraints) -> None:
        """
        Initialize object type model
        Args:
            non_creation_model: Model for non-creation events
            creation_model: Model for creation events
            constraints: Constraints for this object type
        """
        self.obj_type = obj_type
        self.non_creation_model = non_creation_model
        self.creation_model = creation_model
        self.constraints = constraints

    def get_next_scene_distributions(
            self,
            obj_list_prev: ObjList,
            event: Any,
            memory: Optional[StateMemory] = None) -> ObjList:
        """
        Get distributions q_{obj-type} (s'|s, a, memory)
        Args:
            obj_list_prev: s
            event: a
            memory: Memory of past states and events
        """

        existing_objs_dist = self.non_creation_model.get_next_scene_distributions(
            obj_list_prev, event, memory=memory)

        new_objs_dist = self.creation_model.get_next_scene_distributions(
            obj_list_prev, event, memory=memory)

        return ObjList(existing_objs_dist.objs + new_objs_dist.objs, [])

    def sample_next_scene(self,
                          obj_list_prev: ObjList,
                          event: Any,
                          memory: Optional[StateMemory] = None,
                          det: bool = False) -> ObjList:
        """
        Sample from p_{obj-type} (s'|s, a, memory)
        
        Args:
            obj_list_prev: Previous object states
            event: Event/action
            n: Number of samples
            det: Whether to sample deterministically
            
        Returns:
            s' ~ p_{obj-type} (s'|s, a, memory)
        """
        obj_list_dist = self.get_next_scene_distributions(obj_list_prev,
                                                          event,
                                                          memory=memory)

        obj_list = instantiate_obj_list(obj_list_dist, det, temp=0.2)

        return obj_list

    def evaluate_logprobs(self,
                          obj_list_prev: ObjList,
                          event: Any,
                          obj_list_next: ObjList,
                          memory: Optional[StateMemory] = None) -> float:
        non_creation_logprobs = self.non_creation_model.evaluate_logprobs(
            obj_list_prev, event, obj_list_next, memory=memory)
        creation_logprobs = self.creation_model.evaluate_logprobs(
            obj_list_prev, event, obj_list_next, memory=memory)
        return non_creation_logprobs + creation_logprobs

    def prune_programs(self, threshold: float = 0.001) -> None:
        self.non_creation_model.prune_programs(threshold)
        self.creation_model.prune_programs(threshold)

    def clear_cache(self) -> None:
        self.non_creation_model.clear_cache()
        self.creation_model.clear_cache()

    def enable_cache(self) -> None:
        self.non_creation_model.enable_cache()
        self.creation_model.enable_cache()

    def disable_cache(self) -> None:
        self.non_creation_model.disable_cache()
        self.creation_model.disable_cache()

    def clear_precompute_dist(self) -> None:
        self.non_creation_model.clear_precompute_dist()
        self.creation_model.clear_precompute_dist()

    def remove_callables(self) -> 'ObjTypeModel':
        new_non_creation_model = self.non_creation_model.remove_callables()
        new_creation_model = self.creation_model.remove_callables()
        new_constraints = self.constraints.remove_callables()
        return ObjTypeModel(self.obj_type, new_non_creation_model,
                            new_creation_model, new_constraints)

    def prepare_callables(self) -> None:
        self.non_creation_model.prepare_callables()
        self.creation_model.prepare_callables()
        self.constraints.prepare_callables()


class WorldModel(Model):
    """
    Composition of multiple object models
    Combines predictions from multiple models with constraints
    """
    def __init__(self,
                 obj_type_models: List[ObjTypeModel],
                 constraints=None) -> None:
        """
        Initialize composed model
        
        Args:
            obj_models: Sequence of object models to compose
            objects_selector: Function to select relevant objects
            constraints: Optional constraints on predictions
        """
        self.obj_type_models = obj_type_models
        self.obj_types = [x.obj_type for x in obj_type_models]
        self.constraints = constraints

    def get_next_scene_distributions(self,
                                     obj_list_prev: ObjList,
                                     event: Any,
                                     memory: Optional[StateMemory] = None,
                                     mode: str = "all") -> ObjList:
        """
        Get distributions q_{obj-type} (s'|s, a, memory)
        
        Args:
            obj_list_prev: s
            event: a
            memory: Memory of past states and events
            mode: Whether to get distributions for all objects, just player, or non-player objects ("all", "player", "objects")
        """
        if mode == "all":
            obj_types = self.obj_types
        elif mode == "player":
            obj_types = ['player']
        else:
            obj_types = [x for x in self.obj_types if x != 'player']

        objs_lists = []
        for obj_model in self.obj_type_models:
            if obj_model.obj_type in obj_types:
                objs_lists.append(
                    obj_model.get_next_scene_distributions(obj_list_prev,
                                                           event,
                                                           memory=memory))
        return ObjList(sum([objs_list.objs for objs_list in objs_lists], []))

    def _instantiate_obj_list_w_temp(self, obj_list_dist, temp, max_temp, rng,
                                     det):
        """
        Sample obj_list from the distribution with specified temperature
        We are doing this to try to satisfy the constraints
        If done naively, then the sampled obj_list may be very low probability
        Here's the logic:
        - If deterministic, just set deleted and one of velocities to their most likely values
        - If not, sample deleted and one of velocities with temp 0.2
        - The other velocity is sampled with increasing temperature
        Args:
            obj_list_dist: Distribution over objects
            temp: Temperature
            max_temp: Maximum temperature
            rng: Random number generator
            det: Whether to sample some attributes deterministically
        """
        for o in obj_list_dist:
            if o.obj_type == 'player':
                # If deterministic, just set deleted and one of velocities to their most likely values
                # If not, sample deleted and one of velocities with temp 0.2
                # The other velocity is sampled with increasing temperature
                # If temp is high enough, both velocities are sampled with the specified temperature
                if isinstance(o.w_change, RandomValues) or isinstance(
                        o.h_change, RandomValues):
                    raise Exception(
                        'Player objects should not have size changes')

                if det:
                    if temp % 2 == 0 and temp < max_temp * 0.6:
                        o.velocity_x = o.velocity_x.get_max_prob_value()
                    elif temp % 2 == 1 and temp < max_temp * 0.6:
                        o.velocity_y = o.velocity_y.get_max_prob_value()
                    o.deleted = o.deleted.get_max_prob_value()
                else:
                    if temp % 2 == 0 and temp < max_temp * 0.6:
                        o.velocity_x = o.velocity_x.sample(0.2, rng)
                    elif temp % 2 == 1 and temp < max_temp * 0.6:
                        o.velocity_y = o.velocity_y.sample(0.2, rng)
                    o.deleted = o.deleted.sample(0.2, rng)

            else:
                raise Exception(
                    'Should not enter here -- only sample player_obj_list_dist with temp'
                )

        obj_list = instantiate_obj_list(obj_list_dist,
                                        False,
                                        temp=temp,
                                        rng=rng)
        return obj_list

    def _check_constraints(
            self,
            obj_list_prev: ObjList,
            obj_list: ObjList,
            target_n_touchs: Optional[List[int]] = None) -> bool:
        """
        Apply constraints to obj_list
        But need to first construct a full obj list with all obj types in order to apply constraints
        Args:
            obj_list_prev: Full list of previous objects
            obj_list: List of objects only with type self.obj_type
            target_n_touchs: List of object ids that should be touching
        """
        # Constraints need to be applied on full object list
        # For example, we want the player's bottom to align with a platform's top

        # use full list please
        if self.constraints is None:
            return True, []
        full_obj_list = replace_objs_w_specified_types(obj_list_prev,
                                                       obj_list.deepcopy(),
                                                       self.obj_types)
        return self.constraints.apply(full_obj_list,
                                      target_n_touchs=target_n_touchs)

    def _assign_new_pos_and_velocity(self, obj_list_prev: ObjList,
                                     new_obj_list: ObjList) -> ObjList:
        """
        Assign new positions and velocities to objects in obj_list_prev based on new_obj_list
        Args:
            obj_list_prev: Previous object states
            new_obj_list: New object states, with new x, y, and velocities
        """
        obj_list_prev = obj_list_prev.deepcopy()
        pairs, leftover_list1, _ = match_two_obj_lists(obj_list_prev, new_obj_list)
        for i, j in pairs:
            obj_list_prev.objs[i].set_new_pos_and_velocity(
                new_obj_list.objs[j].prev_x, new_obj_list.objs[j].prev_y,
                new_obj_list.objs[j].velocity_x,
                new_obj_list.objs[j].velocity_y)
        for i in leftover_list1:
            obj_list_prev.objs[i].set_new_pos_and_velocity(
                obj_list_prev.objs[i].prev_x, obj_list_prev.objs[i].prev_y,
                obj_list_prev.objs[i].velocity_x,
                obj_list_prev.objs[i].velocity_y)
        return obj_list_prev

    def sample_next_scene(self,
                          obj_list_prev: ObjList,
                          event: Any,
                          memory: Optional[StateMemory] = None,
                          det: bool = False) -> ObjList:
        """
        Sample from p_{obj-type} (s'|s, a, memory) \propto q_{obj-type} (s'|s, a, memory) C(s')
        where C is the constraints for this object type
        
        Args:
            obj_list_prev: Previous object states
            event: Event/action
            n: Number of samples
            det: Whether to sample deterministically
            
        Returns:
            s' ~ p_{obj-type} (s'|s, a, memory)
        """
        

        # Grab non-player objects first
        non_player_obj_list_dist = self.get_next_scene_distributions(
            obj_list_prev, event, memory=memory, mode='objects')
        non_player_obj_list = instantiate_obj_list(non_player_obj_list_dist,
                                                   det,
                                                   temp=0.2)

        # Now sample player object
        obj_list_prev_w_new_pos_and_v = self._assign_new_pos_and_velocity(
            obj_list_prev, non_player_obj_list)
        player_obj_list_dist = self.get_next_scene_distributions(
            obj_list_prev_w_new_pos_and_v, event, memory=memory, mode='player')
        player_obj_list = instantiate_obj_list(player_obj_list_dist,
                                               det,
                                               temp=0.2)

        # Put them together
        obj_list = ObjList(player_obj_list.objs + non_player_obj_list.objs, [])

        # Set this in case we fail to satisfy constraints
        og_obj_list = obj_list

        # Constraints need to be applied on full object list
        # For example, we want the player's bottom to align with a platform's top
        if self.obj_type_models[-1].non_creation_model.config.no_constraints:
            success = True
            target_n_touchs = []
        else:
            success, target_n_touchs = self._check_constraints(
                obj_list_prev, obj_list)

        # If success is False, then the sampled object list needs to satisfy
        # an additional requirement of touching targets in target_n_touchs
        # If we do not enforce this, as we increase the temperature,
        # the sampled object list will just try to avoid all constraints since that tends to be easiest
        # We will try to satisfy this by sampling with increasing temperature

        max_n_tries = 1000

        ct = 1
        rng = np.random.default_rng(0)
        while not success and ct < max_n_tries:
            # grab player obj list with higher temp
            player_obj_list = self._instantiate_obj_list_w_temp(
                player_obj_list_dist.deepcopy(), ct, max_n_tries, rng, det)
            obj_list = ObjList(player_obj_list.objs + non_player_obj_list.objs,
                               [])

            # Try applying constraints again
            success, _ = self._check_constraints(obj_list_prev, obj_list,
                                                 target_n_touchs)

            ct += 1
            rng = np.random.default_rng(ct)

        if ct == max_n_tries:
            obj_list = ObjList([])
            log.warning('Cannot satisfy constraints')

        obj_list = replace_objs_w_specified_types(obj_list_prev, obj_list,
                                                  self.obj_types)
        
        object_tracker = ObjectTracker(init_obj_list=obj_list_prev)
        object_tracker.update(obj_list)

        return obj_list

    def evaluate_logprobs(self,
                          obj_list_prev: ObjList,
                          event: Any,
                          obj_list_next: ObjList,
                          memory: Optional[List[Any]] = None) -> float:
        res = 0
        for obj_model in self.obj_type_models:
            res += obj_model.evaluate_logprobs(obj_list_prev,
                                               event,
                                               obj_list_next,
                                               memory=memory)
        return res

    def get_features(self, obj_list):
        return self.constraints.get_features(obj_list)

    def prune_programs(self, threshold: float = 0.001) -> None:
        for obj_model in self.obj_type_models:
            obj_model.prune_programs(threshold)

    def clear_cache(self) -> None:
        for obj_model in self.obj_type_models:
            obj_model.clear_cache()

    def enable_cache(self) -> None:
        for obj_model in self.obj_type_models:
            obj_model.enable_cache()

    def disable_cache(self) -> None:
        for obj_model in self.obj_type_models:
            obj_model.disable_cache()

    def clear_precompute_dist(self) -> None:
        for obj_model in self.obj_type_models:
            obj_model.clear_precompute_dist()

    def remove_callables(self) -> 'WorldModel':
        new_obj_models = [
            obj_model.remove_callables() for obj_model in self.obj_type_models
        ]
        return WorldModel(new_obj_models, constraints=self.constraints)

    def prepare_callables(self) -> None:
        for x in self.obj_type_models:
            x.prepare_callables()
