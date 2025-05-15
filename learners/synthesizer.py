from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import takewhile
from typing import List, Awaitable

from classes.helper import *
from learners.utils import *
from prompts.synthesizer import *


def get_nl_player_relationships(x,
                                player_interactions_selector,
                                constraints=False):
    output_target_int_list = player_interactions_selector(
        x.output_state.get_obj_interactions())

    observations = []
    axes = []

    for int in output_target_int_list:
        if int.obj1.obj_type == 'player':
            player_obj = int.obj1
            other_obj = int.obj2
        else:
            player_obj = int.obj2
            other_obj = int.obj1

        new_msg = 'new_' if not constraints else ''

        for u_side1, u_side2 in [('center_x', 'center_x'),
                                 ('center_y', 'center_y'),
                                 ('top_side', 'bottom_side')]:
            for side1, side2 in [(u_side1, u_side2), (u_side2, u_side1)]:
                if getattr(player_obj, side1) == getattr(other_obj, side2):
                    observations.append(
                        f"The {player_obj.str_wo_id()} that touches a {other_obj.str_wo_id()} sets its {side1} position to be equal to that {other_obj.str_wo_id()}'s {new_msg}{side2} position"
                    )
                    if side1 in ['center_x', 'left_side', 'right_side']:
                        axes.append('x')
                    else:
                        axes.append('y')
    return observations, axes


class Synthesizer(ABC):
    def __init__(self, config, obj_type, llm):
        self.config = config
        self.obj_type = obj_type
        self.llm = llm

        # Create static variables to be used
        self.objects_selector = ObjTypeObjSelector(self.obj_type)
        self.interactions_selector = ObjTypeInteractionSelector(self.obj_type)
        self.player_interactions_selector = ObjTypeInteractionSelector(
            'player')

        # Create cache
        self.cache_x = {}

    @abstractmethod
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        """
        Synthesizes codes (in text format) based on the given list of
        StateTransitionTriplets
        """
        pass

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   player_int_selector=False):
        observations = []
        to_be_prompteds = []
        for x in c:
            effects = self._get_natural_language_effects(x)

            if len(effects) == 0:
                continue

            input_target_obj_list = self.objects_selector(x.input_state)
            if player_int_selector:
                input_target_int_list = self.player_interactions_selector(
                    x.input_state.get_obj_interactions())
            else:
                input_target_int_list = self.interactions_selector(
                    x.input_state.get_obj_interactions())

            # Construct prompt
            to_be_prompteds.append(
                prompt.format(input=self._prep_interpret_input(
                    input_target_obj_list, input_target_int_list),
                              effects=list_to_bullets(effects)))

        if len(to_be_prompteds) == 0:
            return []

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _get_natural_language_effects(self, x):
        if x not in self.cache_x:
            input_target_obj_list = self.objects_selector(x.input_state)
            output_target_obj_list = self.objects_selector(x.output_state)

            if len(input_target_obj_list) == 0 and len(
                    output_target_obj_list) == 0:
                self.cache_x[x] = []
                return self.cache_x[x]

            effects = []
            input_ids = [o.id for o in input_target_obj_list]
            for o in output_target_obj_list:
                if o.deleted == 1:
                    effects.append(f'The {o.str_w_id()} is deleted')
                elif o.id not in input_ids:
                    effects.append(
                        f'A new {o.obj_type} object is created at ' +
                        f'(x={o.x},y={o.y})')
                else:
                    effects.append(
                        f'The {o.str_w_id()} sets x-axis velocity to ' +
                        f'{"%+d" % (o.velocity_x)}')
                    effects.append(
                        f'The {o.str_w_id()} sets y-axis velocity to ' +
                        f'{"%+d" % (o.velocity_y)}')

            self.cache_x[x] = effects
        return self.cache_x[x]

    def _prep_interpret_input(self, obj_list, int_list=[]):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()},\n'

        txt2 = ',\n'.join([xx.str_w_id() for xx in int_list
                           ]) if len(int_list) > 0 else 'No interactions'
        input_obs = txt1 + '\n' + txt2
        return input_obs

    async def _a_prompts_to_codes_helper(self,
                                         prompts: List[str]) -> List[str]:
        outputs = await self.llm.aprompt(prompts,
                                         temperature=0,
                                         seed=self.config.seed)

        all_rules = []
        for output in outputs:
            rules = process_llm_response_to_codes(output)
            log.debug(f"We got the following rules\n{list_to_str(rules)}")
            all_rules = all_rules + rules
        return all_rules


class ActionSynthesizer(Synthesizer):
    """
    Synthesizing module for action-related events
    """
    async def a_synthesize(self, c: List[StateTransitionTriplet],
                           **kwargs) -> Awaitable[List[str]]:
        action = c[-1].event

        # process c for this module
        c = c[-self.config.synthesizer.synth_window:]

        # if len(
        #         self.interactions_selector(
        #             c[-1].input_state.get_obj_interactions())) == 0:
        #     return []
        prompts, ns = [
            partial_format(interpret_obj_interact_prompt,
                           obj_type=self.obj_type)
        ], [4]
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With action={action}, we see the following observations\n"
                f"{list_to_bullets(observations)}")

        # initial synthesized functions
        to_be_prompteds = [
            explain_event_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                action=action,
                n=n,
            ) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)

        return res


class MultiTimestepActionSynthesizer(Synthesizer):
    """
    Synthesizing module for action-related events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 15
        self.tracking = {}

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        for i in range(1, min(len(c) + 1, self.max_context_length)):
            x = c[-i]
            action = x.event
            if action != 'RESTART' and len(
                    self.interactions_selector(
                        x.input_state.get_obj_interactions())) != 0:
                if i == 1: # ActionSynthesizer already handles this
                    return []
                if action == 'NOOP': # NOOP is an action action -- let POMDPVelocity handle this
                    return []
                prompts, ns = [
                    partial_format(interpret_obj_interact_pomdp_prompt,
                                   obj_type=self.obj_type)
                ], [4]
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(c[-i:], prompt)
                    for prompt in prompts
                ])

                to_be_prompteds = [
                    explain_event_pomdp_prompt.format(
                        obj_type=self.obj_type,
                        obs_lst_txt=list_to_bullets(observations),
                        action=action,
                        n=n,
                    ) for observations, n in zip(observations_list, ns)
                    if len(observations) > 0
                ]

                res = await self._a_prompts_to_codes_helper(to_be_prompteds)
                # turn res into a list of tuples with the second element being i
                res = [(x, i) for x in res]
                return res  # Stop as soon as found a potential cause -- not sure if this is a good idea

        return []

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   player_int_selector=False):
        observations = []
        to_be_prompteds = []

        x = c[0]

        effects = self._get_natural_language_effects(c)

        input_target_obj_list = self.objects_selector(x.input_state)
        if player_int_selector:
            input_target_int_list = self.player_interactions_selector(
                x.input_state.get_obj_interactions())
        else:
            input_target_int_list = self.interactions_selector(
                x.input_state.get_obj_interactions())

        # Construct prompt
        to_be_prompteds.append(
            prompt.format(input=self._prep_interpret_input(
                input_target_obj_list, input_target_int_list),
                          effects=list_to_bullets(effects)))

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _get_natural_language_effects(self, c):
        deleted_lst = []
        velocity_x_lst = []
        velocity_y_lst = []

        input_target_obj_list = self.objects_selector(c[0].input_state)

        # Skip if we can't find target object in c[0]
        if len(input_target_obj_list) == 0:
            return []

        if len(input_target_obj_list) != 1:
            # raise NotImplementedError("Only support one object for now")
            return []

        input_id = input_target_obj_list[0].id

        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            if input_id not in output_ids:
                velocity_x_lst.append(0)
                velocity_y_lst.append(0)
                deleted_lst.append(1)
            else:
                for o in output_target_obj_list:
                    if o.id == input_id:
                        velocity_x_lst.append(o.velocity_x)
                        velocity_y_lst.append(o.velocity_y)
                        deleted_lst.append(0)
                        break

        velocity_x_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_x_lst])
        velocity_y_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_y_lst])
        deleted_lst_txt = ', '.join([f'{"%+d" % x}' for x in deleted_lst])
        effects = [
            f'The {o.str_w_id()} sets x-axis velocity to [{velocity_x_lst_txt}]',
            f'The {o.str_w_id()} sets y-axis velocity to [{velocity_y_lst_txt}]',
            f'The {o.str_w_id()} sets its visibility to [{deleted_lst_txt}]',
        ]
        return effects


class MultiTimestepMomentumSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 10

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        all_res = []

        # Velocity x
        objs = self.objects_selector(c[-1].output_state)
        if objs[0].velocity_x != 0:
            velocity_dir = objs[0].velocity_x
            for i in range(min(len(c), self.max_context_length - 1)):
                x = c[-i]
                objs = self.objects_selector(x.input_state)
                # if it's zero, keep looking
                if len(objs) == 0 or objs[0].velocity_x == 0:
                    continue
                
                if velocity_dir * objs[0].velocity_x > 0:  # no momentum change
                    break

                if velocity_dir * objs[0].velocity_x < 0:  # momentum change
                    prompts, ns = [
                        partial_format(interpret_obj_momentum_pomdp_x_prompt,
                                    obj_type=self.obj_type,
                                    axis='x-axis')
                    ], [1]
                    observations_list = await asyncio.gather(*[
                        self._a_get_natural_language_observations(
                            c[-i:], prompt, 'x-axis') for prompt in prompts
                    ])

                    if len(observations_list) != 0:
                        to_be_prompteds = [
                            explain_event_passive_pomdp_prompt.format(
                                obj_type=self.obj_type,
                                obs_lst_txt=list_to_bullets(observations),
                                n=n)
                            for observations, n in zip(observations_list, ns)
                            if len(observations) > 0
                        ]

                        res = await self._a_prompts_to_codes_helper(to_be_prompteds
                                                                    )
                        # turn res into a list of tuples with the second element being i
                        res = [(x, i) for x in res]

                        all_res = all_res + res
                    break

        # Velocity y
        objs = self.objects_selector(c[-1].output_state)
        if objs[0].velocity_y != 0:
            velocity_dir = objs[0].velocity_y
            for i in range(min(len(c), self.max_context_length - 1)):
                x = c[-i]
                objs = self.objects_selector(x.input_state)
                # if it's zero, keep looking

                if len(objs) == 0 or objs[0].velocity_y == 0:
                    continue

                if velocity_dir * objs[0].velocity_y > 0:  # no momentum change
                    break

                if velocity_dir * objs[0].velocity_y < 0:  # momentum change
                    prompts, ns = [
                        partial_format(interpret_obj_momentum_pomdp_y_prompt,
                                    obj_type=self.obj_type,
                                    axis='y-axis')
                    ], [1]
                    observations_list = await asyncio.gather(*[
                        self._a_get_natural_language_observations(
                            c[-i:], prompt, 'y-axis') for prompt in prompts
                    ])

                    if len(observations_list) != 0:
                        to_be_prompteds = [
                            explain_event_passive_pomdp_prompt.format(
                                obj_type=self.obj_type,
                                obs_lst_txt=list_to_bullets(observations),
                                n=n)
                            for observations, n in zip(observations_list, ns)
                            if len(observations) > 0
                        ]

                        res = await self._a_prompts_to_codes_helper(to_be_prompteds
                                                                    )
                        # turn res into a list of tuples with the second element being i
                        res = [(x, i) for x in res]

                        all_res = all_res + res
                    break

        return all_res

    # From MultiTimestepActionSynthesizer
    async def _a_get_natural_language_observations(self, c, prompt, mode):
        observations = []
        to_be_prompteds = []

        x = c[0]

        effects = self._get_natural_language_effects(c, mode)

        input_target_obj_list = self.objects_selector(x.input_state)

        # Construct prompt
        to_be_prompteds.append(
            prompt.format(input=self._prep_interpret_input(
                input_target_obj_list, mode),
                          effects=list_to_bullets(effects)))

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    # From MultiTimestepActionSynthesizer
    def _get_natural_language_effects(self, c, mode):
        deleted_lst = []
        velocity_x_lst = []
        velocity_y_lst = []

        input_target_obj_list = self.objects_selector(c[0].input_state)

        # Skip if we can't find target object in c[0]
        if len(input_target_obj_list) == 0:
            return []

        if len(input_target_obj_list) != 1:
            # raise NotImplementedError("Only support one object for now")
            return []

        input_id = input_target_obj_list[0].id

        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            if input_id not in output_ids:
                velocity_x_lst.append(0)
                velocity_y_lst.append(0)
                deleted_lst.append(1)
            else:
                for o in output_target_obj_list:
                    if o.id == input_id:
                        velocity_x_lst.append(o.velocity_x)
                        velocity_y_lst.append(o.velocity_y)
                        deleted_lst.append(0)
                        break

        velocity_x_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_x_lst])
        velocity_y_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_y_lst])
        deleted_lst_txt = ', '.join([f'{"%+d" % x}' for x in deleted_lst])
        if mode == 'x-axis':
            effects = [
                f'The {o.str_w_id()} sets x-axis velocity to [{velocity_x_lst_txt}]'
            ]
        elif mode == 'y-axis':
            effects = [
                f'The {o.str_w_id()} sets y-axis velocity to [{velocity_y_lst_txt}]'
            ]
        else:
            raise NotImplementedError
        return effects

    def _prep_interpret_input(self, obj_list, mode):
        txt1 = ''
        for obj in obj_list:
            if mode == 'x-axis':
                txt1 += f'{obj.str_w_id()}' + f' with x-axis velocity = {"%+d" % obj.velocity_x}\n'
                txt1 += f'{obj.str_w_id()}' + f' is at x={obj.x}\n'
            elif mode == 'y-axis':
                txt1 += f'{obj.str_w_id()}' + f' with y-axis velocity = {"%+d" % obj.velocity_y}\n'
                txt1 += f'{obj.str_w_id()}' + f' is at y={obj.y}\n'
            else:
                raise NotImplementedError

        # Ignore int_list
        input_obs = txt1
        return input_obs


class MultiTimestepSizeChangeSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 60

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        size_change_dir = None
        for i in range(min(len(c), self.max_context_length - 1)):
            x = c[-i]
            objs = self.objects_selector(x.input_state)
            if len(objs) == 0 or objs[0].w_change == 0:
                continue

            if size_change_dir is None:
                size_change_dir = 1 if objs[0].w_change > 0 else -1
                continue

            if size_change_dir * objs[0].w_change < 0:  # momentum change
                prompts, ns = [
                    partial_format(interpret_obj_size_change_pomdp_prompt,
                                   obj_type=self.obj_type)
                ], [2]
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(c[-i:], prompt)
                    for prompt in prompts
                ])

                if len(observations_list) == 0:
                    return []

                to_be_prompteds = [
                    explain_event_passive_pomdp_2_prompt.format(
                        obj_type=self.obj_type,
                        obs_lst_txt=list_to_bullets(observations),
                        n=n) for observations, n in zip(observations_list, ns)
                    if len(observations) > 0
                ]

                res = await self._a_prompts_to_codes_helper(to_be_prompteds)
                # turn res into a list of tuples with the second element being i
                res = [(x, i) for x in res]
                return res  # Stop as soon as found a potential cause -- not sure if this is a good idea

        return []

    def _shorten_list_txt(self, lst):
        return lst
        # max_zero_start, max_zero_len = -1, 0
        # i = 0
        # while i < len(lst):
        #     if lst[i] == 0:
        #         start = i
        #         while i < len(lst) and lst[i] == 0:
        #             i += 1
        #         count = i - start
        #         if count > max_zero_len:
        #             max_zero_start, max_zero_len = start, count
        #     else:
        #         i += 1

        # if max_zero_len > 3:
        #     return str(lst[:max_zero_start]) + f" + [0] * {max_zero_len} + " + str(lst[max_zero_start + max_zero_len:])
        # return str(lst)

    # From MultiTimestepActionSynthesizer
    async def _a_get_natural_language_observations(self, c, prompt):
        observations = []
        to_be_prompteds = []

        x = c[0]

        effects = self._get_natural_language_effects(c)

        input_target_obj_list = self.objects_selector(x.input_state)

        # Construct prompt
        to_be_prompteds.append(
            prompt.format(
                input=self._prep_interpret_input(input_target_obj_list),
                effects=list_to_bullets(effects)))

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    # From MultiTimestepActionSynthesizer
    def _get_natural_language_effects(self, c):
        deleted_lst = []
        w_change_lst = []
        h_change_lst = []

        input_target_obj_list = self.objects_selector(c[0].input_state)

        # Skip if we can't find target object in c[0]
        if len(input_target_obj_list) == 0:
            return []

        if len(input_target_obj_list) != 1:
            # raise NotImplementedError("Only support one object for now")
            return []

        input_id = input_target_obj_list[0].id

        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            if input_id not in output_ids:
                w_change_lst.append(0)
                h_change_lst.append(0)
                deleted_lst.append(1)
            else:
                for o in output_target_obj_list:
                    if o.id == input_id:
                        w_change_lst.append(o.w_change)
                        h_change_lst.append(o.h_change)
                        deleted_lst.append(0)
                        break

        w_change_lst_txt = self._shorten_list_txt(w_change_lst)
        h_change_lst_txt = self._shorten_list_txt(h_change_lst)
        effects = [
            f'The {o.str_w_id()} changes its width change rate to [{w_change_lst_txt}]',
            f'The {o.str_w_id()} changes its height change rate to [{h_change_lst_txt}]'
        ]
        return effects

    def _prep_interpret_input(self, obj_list):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()}' + f' with width change rate = {"%+d" % obj.w_change}' \
                + f' with height change rate = {"%+d" % obj.h_change}\n'
            txt1 += f'{obj.str_w_id()}' + f' has its size equal to (w={obj.w},h={obj.h})\n'

        # Ignore int_list
        input_obs = txt1
        return input_obs


class MultiTimestepStatusChangeSynthesizer(Synthesizer):
    """
    Synthesizing module for status-change-related events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 35
        self.tracking = {}

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        for i in range(2, min(len(c) + 1, self.max_context_length)):
            x = c[-i]
            objs = self.objects_selector(x.output_state)
            if len(objs) > 0 and objs[0].history['deleted'][-1] != objs[
                    0].history['deleted'][-2]:
                prompts, ns = [
                    partial_format(interpret_obj_interact_pomdp_prompt,
                                   obj_type=self.obj_type)
                ], [2]  # TODO: Delete -- there is no prompting
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(c[-i:], prompt)
                    for prompt in prompts
                ])

                to_be_prompteds = [
                    explain_status_event_pomdp_prompt.format(
                        action='becoming visible'
                        if objs[0].history['deleted'][-1] == 0 else
                        'disappearing',
                        obj_type=self.obj_type,
                        obs_lst_txt=list_to_bullets(observations),
                        n=n,
                    ) for observations, n in zip(observations_list, ns)
                    if len(observations) > 0
                ]

                if len(to_be_prompteds) == 0:
                    return []

                res = await self._a_prompts_to_codes_helper(to_be_prompteds)
                # turn res into a list of tuples with the second element being i
                res = [(x, i - 1) for x in res]
                return res  # Stop as soon as found a potential cause -- not sure if this is a good idea

        return []

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   player_int_selector=False):
        effects = self._get_natural_language_effects(c)

        res = effects
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _get_natural_language_effects(self, c):
        deleted_lst_by_id = defaultdict(list)
        input_target_obj_list = self.objects_selector(c[0].output_state)

        # Since we're detecting deletion with c[0].output_state -- we should start at index 1 instead 0
        for x in c[1:]:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            for o in input_target_obj_list:
                if o.id not in output_ids:
                    deleted_lst_by_id[o.id].append(1)
                else:
                    deleted_lst_by_id[o.id].append(
                        output_target_obj_list[output_ids.index(o.id)].deleted)

        effects = []
        for input_id in deleted_lst_by_id:
            deleted_lst = deleted_lst_by_id[input_id]
            len_prefix = sum(
                1
                for _ in takewhile(lambda x: x == deleted_lst[0], deleted_lst))
            len_suffix = sum(1 for _ in takewhile(
                lambda x: x == deleted_lst[-1], reversed(deleted_lst)))
            if len_prefix == len(deleted_lst):
                effects.append(
                    f'The {o.str_wo_id()} (id = {input_id}) sets its visibility to [{deleted_lst[0]}] * {len(deleted_lst)}'
                )
            elif len_prefix >= len_suffix:
                deleted_lst_txt = ', '.join(
                    [str(x) for x in deleted_lst[-len_suffix:]])
                effects.append(
                    f'The {o.str_wo_id()} (id = {input_id}) sets its visibility to [{deleted_lst[0]}] * {len_prefix} + [{deleted_lst_txt}]'
                )
            else:
                deleted_lst_txt = ', '.join(
                    [str(x) for x in deleted_lst[:len_prefix]])
                effects.append(
                    f'The {o.str_wo_id()} (id = {input_id}) sets its visibility to [{deleted_lst_txt}] + [{deleted_lst[-1]}] * {len_suffix}'
                )

        almost_last_target_obj_list = self.objects_selector(c[-1].input_state)
        almost_last_ids = [o.id for o in almost_last_target_obj_list]
        last_target_obj_list = self.objects_selector(c[-1].output_state)
        for o in last_target_obj_list:
            if o.id not in almost_last_ids:
                effects.append(
                    f'A new {o.obj_type} object is created at (x={o.x},y={o.y}) after {len(c) - 1} timesteps'
                )

        return effects


class MultiTimestepStatusChangeVelocityModeSynthesizer(Synthesizer):
    """
    Synthesizing module for status-change-related events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 10
        self.tracking = {}

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        for i in range(1, min(len(c) + 1, self.max_context_length)):
            x = c[-i]
            objs = self.objects_selector(x.input_state)
            if len(objs) > 0 and objs[0].history['deleted'][-1] != objs[
                    0].history['deleted'][-2]:
                prompts, ns = [
                    partial_format(interpret_obj_interact_pomdp_prompt,
                                   obj_type=self.obj_type)
                ], [2]  # TODO: Delete -- there is no prompting
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(c[-i:], prompt)
                    for prompt in prompts
                ])

                to_be_prompteds = [
                    explain_status_event_pomdp_prompt.format(
                        action='becoming visible'
                        if objs[0].history['deleted'][-1] == 0 else
                        'disappearing',
                        obj_type=self.obj_type,
                        obs_lst_txt=list_to_bullets(observations),
                        n=n,
                    ) for observations, n in zip(observations_list, ns)
                    if len(observations) > 0
                ]

                if len(to_be_prompteds) == 0:
                    return []

                res = await self._a_prompts_to_codes_helper(to_be_prompteds)
                # turn res into a list of tuples with the second element being i
                res = [(x, i - 1) for x in res]
                return res  # Stop as soon as found a potential cause -- not sure if this is a good idea

        return []

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   player_int_selector=False):
        effects = self._get_natural_language_effects(c)

        res = effects
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _shorten_list_txt(self, lst):
        max_zero_start, max_zero_len = -1, 0
        i = 0
        while i < len(lst):
            if lst[i] == 0:
                start = i
                while i < len(lst) and lst[i] == 0:
                    i += 1
                count = i - start
                if count > max_zero_len:
                    max_zero_start, max_zero_len = start, count
            else:
                i += 1

        if max_zero_len > 0:
            return str(
                lst[:max_zero_start]) + f" + [0] * {max_zero_len} + " + str(
                    lst[max_zero_start + max_zero_len:])
        return str(lst)

    def _get_natural_language_effects(self, c):
        velocity_x_lst_by_id = defaultdict(list)
        velocity_y_lst_by_id = defaultdict(list)
        input_target_obj_list = self.objects_selector(c[0].output_state)

        # Since we're detecting deletion with c[0].output_state -- we should start at index 1 instead 0
        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            for o in input_target_obj_list:
                velocity_x_lst_by_id[o.id].append(
                    output_target_obj_list[output_ids.index(o.id)].velocity_x)
                velocity_y_lst_by_id[o.id].append(
                    output_target_obj_list[output_ids.index(o.id)].velocity_y)

        effects = []
        for input_id in velocity_x_lst_by_id:
            velocity_x_lst = velocity_x_lst_by_id[input_id]
            velocity_y_lst = velocity_y_lst_by_id[input_id]
            velocity_x_lst_txt = self._shorten_list_txt(velocity_x_lst)
            velocity_y_lst_txt = self._shorten_list_txt(velocity_y_lst)
            effects.append(
                f'The {o.str_wo_id()} sets x-axis velocity to {velocity_x_lst_txt}'
            )
            effects.append(
                f'The {o.str_wo_id()} sets y-axis velocity to {velocity_y_lst_txt}'
            )

        return effects


class MultiTimestepStatusChangeSizeModeSynthesizer(Synthesizer):
    """
    Synthesizing module for status-change-related events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 60
        self.tracking = {}

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        for i in range(1, min(len(c) + 1, self.max_context_length)):
            x = c[-i]
            objs = self.objects_selector(x.input_state)
            if len(objs) > 0 and objs[0].history['deleted'][-1] != objs[
                    0].history['deleted'][-2]:
                prompts, ns = [
                    partial_format(interpret_obj_interact_pomdp_prompt,
                                   obj_type=self.obj_type)
                ], [2]  # TODO: Delete -- there is no prompting
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(c[-i:], prompt)
                    for prompt in prompts
                ])

                to_be_prompteds = [
                    explain_status_event_pomdp_2_prompt.format(
                        action='becoming visible'
                        if objs[0].history['deleted'][-1] == 0 else
                        'disappearing',
                        obj_type=self.obj_type,
                        obs_lst_txt=list_to_bullets(observations),
                        n=n,
                    ) for observations, n in zip(observations_list, ns)
                    if len(observations) > 0
                ]

                if len(to_be_prompteds) == 0:
                    return []

                res = await self._a_prompts_to_codes_helper(to_be_prompteds)
                # turn res into a list of tuples with the second element being i
                res = [(x, i - 1) for x in res]
                return res  # Stop as soon as found a potential cause -- not sure if this is a good idea

        return []

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   player_int_selector=False):
        effects = self._get_natural_language_effects(c)

        res = effects
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _shorten_list_txt(self, lst):
        max_zero_start, max_zero_len = -1, 0
        i = 0
        while i < len(lst):
            if lst[i] == 0:
                start = i
                while i < len(lst) and lst[i] == 0:
                    i += 1
                count = i - start
                if count > max_zero_len:
                    max_zero_start, max_zero_len = start, count
            else:
                i += 1

        if max_zero_len > 0:
            return str(
                lst[:max_zero_start]) + f" + [0] * {max_zero_len} + " + str(
                    lst[max_zero_start + max_zero_len:])
        return str(lst)

    def _get_natural_language_effects(self, c):
        w_change_lst_by_id = defaultdict(list)
        h_change_lst_by_id = defaultdict(list)
        input_target_obj_list = self.objects_selector(c[0].output_state)

        # Since we're detecting deletion with c[0].output_state -- we should start at index 1 instead 0
        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            for o in input_target_obj_list:
                w_change_lst_by_id[o.id].append(
                    output_target_obj_list[output_ids.index(o.id)].w_change)
                h_change_lst_by_id[o.id].append(
                    output_target_obj_list[output_ids.index(o.id)].h_change)

        effects = []
        for input_id in w_change_lst_by_id:
            w_change_lst = w_change_lst_by_id[input_id]
            h_change_lst = h_change_lst_by_id[input_id]
            w_change_lst_txt = self._shorten_list_txt(w_change_lst)
            h_change_lst_txt = self._shorten_list_txt(h_change_lst)
            effects.append(
                f'The {o.str_wo_id()} sets its width change rate to {w_change_lst_txt}'
            )
            effects.append(
                f'The {o.str_wo_id()} sets its height change rate to {h_change_lst_txt}'
            )

        return effects


class PassiveMovementSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        # process c for this module
        c = c[-self.config.synthesizer.synth_window:]

        prompts, ns = ([
            partial_format(interpret_velocity_prompt, obj_type=self.obj_type)
        ], [6])
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With passive movement, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_passive_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        if len(to_be_prompteds) == 0:
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res

    def _prep_interpret_input(self, obj_list, int_list=[]):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()}' + f' with x-axis velocity = {"%+d" % obj.velocity_x}' \
                + f' with y-axis velocity = {"%+d" % obj.velocity_y}\n'

        txt2 = ',\n'.join([xx.str_w_id() for xx in int_list
                           ]) if len(int_list) > 0 else 'No interactions'
        input_obs = txt1 + '\n' + txt2
        return input_obs
    
class PassiveCreationSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        # process c for this module
        c = c[-self.config.synthesizer.synth_window:]

        prompts, ns = ([
            partial_format(interpret_velocity_creation_prompt, obj_type=self.obj_type)
        ], [4])
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With passive movement, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_passive_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        if len(to_be_prompteds) == 0:
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res
    

class VelocitySynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        # process c for this module
        c = [x for x in c if x.output_game_state != GameState.RESTART]
        c = c[-self.config.synthesizer.synth_window:]

        prompts, ns = ([
            partial_format(interpret_velocity_4_prompt, obj_type=self.obj_type)
        ], [4])
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With passive movement, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_passive_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        if len(to_be_prompteds) == 0:
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res

    def _prep_interpret_input(self, obj_list, int_list=[]):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()}' + f' with x-axis velocity = {"%+d" % obj.velocity_x}' \
                + f' with y-axis velocity = {"%+d" % obj.velocity_y}\n'

        txt2 = ',\n'.join([xx.str_w_id() for xx in int_list
                           ]) if len(int_list) > 0 else 'No interactions'
        input_obs = txt1 + '\n' + txt2
        return input_obs


class MultiTimestepVelocitySynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement events in POMDP
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.max_context_length = 5

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        all_res = []

        # Velocity x
        objs = self.objects_selector(c[-1].output_state)
        if objs[0].velocity_x != 0:
            for i in range(1, min(len(c), self.max_context_length - 1)):
                x = c[-i]
                objs = self.objects_selector(x.input_state)
                # if it's zero, keep looking
                if len(objs) == 0 or objs[0].velocity_x == 0:
                    continue
                
                if i == 1:
                    break
                
                prompts, ns = [
                    partial_format(interpret_obj_velocity_pomdp_x_prompt,
                                obj_type=self.obj_type,
                                axis='x-axis')
                ], [2]
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(
                        c[-i:], prompt, 'x-axis') for prompt in prompts
                ])

                if len(observations_list) != 0:
                    to_be_prompteds = [
                        explain_event_passive_pomdp_prompt.format(
                            obj_type=self.obj_type,
                            obs_lst_txt=list_to_bullets(observations),
                            n=n)
                        for observations, n in zip(observations_list, ns)
                        if len(observations) > 0
                    ]

                    res = await self._a_prompts_to_codes_helper(to_be_prompteds
                                                                )
                    # turn res into a list of tuples with the second element being i
                    res = [(x, i) for x in res]

                    all_res = all_res + res
                break

        # Velocity y
        objs = self.objects_selector(c[-1].output_state)
        if objs[0].velocity_y != 0:
            for i in range(1, min(len(c), self.max_context_length - 1)):
                x = c[-i]
                objs = self.objects_selector(x.input_state)
                # if it's zero, keep looking

                if len(objs) == 0 or objs[0].velocity_y == 0:
                    continue
                
                if i == 1:
                    break

                prompts, ns = [
                    partial_format(interpret_obj_velocity_pomdp_y_prompt,
                                obj_type=self.obj_type,
                                axis='y-axis')
                ], [2]
                observations_list = await asyncio.gather(*[
                    self._a_get_natural_language_observations(
                        c[-i:], prompt, 'y-axis') for prompt in prompts
                ])

                if len(observations_list) != 0:
                    to_be_prompteds = [
                        explain_event_passive_pomdp_prompt.format(
                            obj_type=self.obj_type,
                            obs_lst_txt=list_to_bullets(observations),
                            n=n)
                        for observations, n in zip(observations_list, ns)
                        if len(observations) > 0
                    ]

                    res = await self._a_prompts_to_codes_helper(to_be_prompteds
                                                                )
                    # turn res into a list of tuples with the second element being i
                    res = [(x, i) for x in res]

                    all_res = all_res + res
                break

        return all_res

    # From MultiTimestepActionSynthesizer
    async def _a_get_natural_language_observations(self, c, prompt, mode):
        observations = []
        to_be_prompteds = []

        x = c[0]

        effects = self._get_natural_language_effects(c, mode)

        input_target_obj_list = self.objects_selector(x.input_state)
        
        input_target_int_list = self.interactions_selector(
                    x.input_state.get_obj_interactions())

        # Construct prompt
        to_be_prompteds.append(
            prompt.format(input=self._prep_interpret_input(
                input_target_obj_list, input_target_int_list, mode),
                          effects=list_to_bullets(effects)))

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    # From MultiTimestepActionSynthesizer
    def _get_natural_language_effects(self, c, mode):
        deleted_lst = []
        velocity_x_lst = []
        velocity_y_lst = []

        input_target_obj_list = self.objects_selector(c[0].input_state)

        # Skip if we can't find target object in c[0]
        if len(input_target_obj_list) == 0:
            return []

        if len(input_target_obj_list) != 1:
            # raise NotImplementedError("Only support one object for now")
            return []

        input_id = input_target_obj_list[0].id

        for x in c:
            output_target_obj_list = self.objects_selector(x.output_state)
            output_ids = [o.id for o in output_target_obj_list]

            if input_id not in output_ids:
                velocity_x_lst.append(0)
                velocity_y_lst.append(0)
                deleted_lst.append(1)
            else:
                for o in output_target_obj_list:
                    if o.id == input_id:
                        velocity_x_lst.append(o.velocity_x)
                        velocity_y_lst.append(o.velocity_y)
                        deleted_lst.append(0)
                        break

        velocity_x_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_x_lst])
        velocity_y_lst_txt = ', '.join(
            [f'{"%+d" % x}' for x in velocity_y_lst])
        deleted_lst_txt = ', '.join([f'{"%+d" % x}' for x in deleted_lst])
        if mode == 'x-axis':
            effects = [
                f'The {o.str_w_id()} sets x-axis velocity to [{velocity_x_lst_txt}]'
            ]
        elif mode == 'y-axis':
            effects = [
                f'The {o.str_w_id()} sets y-axis velocity to [{velocity_y_lst_txt}]'
            ]
        else:
            raise NotImplementedError
        return effects

    def _prep_interpret_input(self, obj_list, int_list, mode):
        txt1 = ''
        for obj in obj_list:
            if mode == 'x-axis':
                txt1 += f'{obj.str_w_id()}' + f' with x-axis velocity = {"%+d" % obj.velocity_x}\n'
            elif mode == 'y-axis':
                txt1 += f'{obj.str_w_id()}' + f' with y-axis velocity = {"%+d" % obj.velocity_y}\n'
            else:
                raise NotImplementedError

        txt2 = ',\n'.join([xx.str_w_id() for xx in int_list
                           ]) if len(int_list) > 0 else 'No interactions'
        input_obs = txt1 + '\n' + txt2
        return input_obs


class VelocityTrackingSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement-related events
    """
    def __init__(self, *args):
        super().__init__(*args)
        if self.obj_type != 'player':
            raise Exception('This module only works on player objects')

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        # process c for this module
        c = [x for x in c if x.output_game_state != GameState.RESTART]
        c = c[-self.config.synthesizer.synth_window:]

        action = c[-1].event

        prompts, ns = ([
            partial_format(interpret_velocity_3_prompt, obj_type=self.obj_type)
        ], [2])
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With passive movement, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_v_tracking_prompt.format(
                action=action,
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        if len(to_be_prompteds) == 0:
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res

    def _prep_interpret_input(self, obj_list, int_list):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()}\n'

        # Filter out object that's about to get deleted
        int_list = [
            x for x in int_list if x.obj1.obj_type == 'player'
            and x.obj2.new_velocity_x is not None or
            x.obj2.obj_type == 'player' and x.obj1.new_velocity_x is not None
        ]

        txt1 += ',\n'.join([xx.str_w_id() for xx in int_list
                            ]) if len(int_list) > 0 else 'No interactions'
        for int in int_list:
            if int.obj1.obj_type == 'player':
                obj = int.obj2
            else:
                obj = int.obj1

            txt1 += f'{obj.str_w_id()}' + f' with new x-axis velocity = {"%+d" % obj.new_velocity_x}' \
                + f' with new y-axis velocity = {"%+d" % obj.new_velocity_y}\n'

        input_obs = txt1
        return input_obs


class NoInteractPassiveMovementSynthesizer(Synthesizer):
    """
    Synthesizing module for passive movement-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        # process c for this module
        c = [x for x in c if x.output_game_state != GameState.RESTART]
        c = c[-self.config.synthesizer.synth_window:]

        prompts, ns = ([
            partial_format(interpret_no_int_prompt, obj_type=self.obj_type)
        ], [4])
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(c, prompt)
            for prompt in prompts
        ])

        for observations in observations_list:
            log.debug(
                f"With passive movement, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_passive_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]

        if len(to_be_prompteds) == 0:
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res

    def _prep_interpret_input(self, obj_list, int_list=[]):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()},\n'

        # Ignore int_list
        input_obs = txt1
        return input_obs


class PlayerInteractionSynthesizer(Synthesizer):
    """
    Synthesizing module for HUD item-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        x = c[-1]

        prompts, ns = [
            partial_format(interpret_5_prompt, obj_type=self.obj_type)
        ], [4]

        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(
                [x], prompt, player_int_selector=True) for prompt in prompts
        ])
        for observations in observations_list:
            log.debug(
                f"For HUD item, we see the following observations\n{list_to_bullets(observations)}"
            )

        to_be_prompteds = [
            explain_event_prompt.format(
                obj_type=self.obj_type,
                action=x.event,
                obs_lst_txt=list_to_bullets(observations),
                n=n) for observations, n in zip(observations_list, ns)
        ]

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)

        return res


class SnappingSynthesizer(Synthesizer):
    """
    Synthesizing module for snapping-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        x = c[-1]
        observations, axes = get_nl_player_relationships(
            x, self.player_interactions_selector)

        if len(observations) == 0:
            return []

        to_be_prompteds = [
            explain_event_snapping_prompt.format(obj_type=self.obj_type,
                                                 obs_lst_txt=observation,
                                                 action=x.event,
                                                 n=1)
            for observation in observations
        ]

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res


class ConstraintsSynthesizer(Synthesizer):
    """
    Synthesizing module for constraints-related events
    """
    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        x = c[-1]

        observations, axes = get_nl_player_relationships(
            x, self.player_interactions_selector, constraints=True)

        if len(observations) == 0:
            return []

        to_be_prompteds = [
            explain_event_constraints_prompt.format(obj_type=self.obj_type,
                                                    obs_lst_txt=observation,
                                                    axis=axis)
            for observation, axis in zip(observations, axes)
        ]

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)
        return res


class RestartSynthesizer(Synthesizer):
    """
    Synthesizing module for restart-related events
    """
    def __init__(self, config, obj_type, llm):
        super().__init__(config, obj_type, llm)
        self.program_name = 'compute_danger_attribute'

    async def a_synthesize(self, c: list[StateTransitionTriplet], **kwargs):
        x = c[-1]

        compute_att_callables, compute_att_txts = await self._a_grab_compute_att_callables(
            x)

        prompts, ns = [
            partial_format(interpret_4_prompt, obj_type=self.obj_type)
        ], [4]
        observations_list = await asyncio.gather(*[
            self._a_get_natural_language_observations(
                c[-1:], prompt, compute_att_callable) for prompt in prompts
            for compute_att_callable in compute_att_callables
        ])
        ns = ns * len(observations_list)

        # initial synthesized functions
        to_be_prompteds = [
            explain_event_passive_danger_prompt.format(
                obj_type=self.obj_type,
                obs_lst_txt=list_to_bullets(observations),
                n=n,
            ) for observations, n in zip(observations_list, ns)
            if len(observations) > 0
        ]
        
        if len(to_be_prompteds) == 0:
            log.warning(f'No restart... ')
            return []

        res = await self._a_prompts_to_codes_helper(to_be_prompteds)

        # repeated_compute_att_txts = [txt for txt in compute_att_txts for _ in range(ns[0])]
        # res = [compute_att_txt + '\n\n' + x for compute_att_txt, x in zip(repeated_compute_att_txts, res)]
        repeated_compute_att_txts = sum(
            [[self._add_tabs_to_string(txt)] * ns[0]
             for txt in compute_att_txts], [])
        
        final_res = []
        for compute_att_txt, x in zip(repeated_compute_att_txts, res):
            while not x.split('\n', 1)[0].startswith(f'def '):
                x = x.split('\n', 1)[1]
            final_res.append(x.split('\n', 1)[0] + '\n' + compute_att_txt + '\n\n' +
            x.split('\n', 1)[1])

        return final_res

    def _add_tabs_to_string(self, input_string, num_spaces=4):
        tab = ' ' * num_spaces
        lines = input_string.split('\n')
        tabbed_lines = [tab + line for line in lines]
        return '\n'.join(tabbed_lines)

    def _player_history_txt(self, obj):
        return f"""\
- The player object has 
    history['velocity_x'] = {obj.history['velocity_x'][-5:]}
    history['velocity_y'] = {obj.history['velocity_y'][-5:]}
"""

    def _process_func_name(self, txt):
        txt = txt[:len(f'def {self.program_name}')] + txt[txt.find('(obj: Obj)'
                                                                   ):]
        return txt

    async def _a_grab_compute_att_callables(self, x):
        obj_list = x.input_state
        player_obj = self.objects_selector(obj_list)[0]

        outputs = await self.llm.aprompt([
            danger_att_prompt.format(
                lst_txt=self._player_history_txt(player_obj))
        ],
                                         temperature=0,
                                         seed=self.config.seed)

        log.info('Prompt')
        log.info(
            danger_att_prompt.format(
                lst_txt=self._player_history_txt(player_obj)))
        log.info('LLM output')
        log.info(outputs[0])

        att_txts = process_llm_response_to_codes(outputs[0])
        # remove _i from name_i
        att_txts = [self._process_func_name(att_txt) for att_txt in att_txts]

        compute_att_callables = []
        for att_txt in att_txts:
            # Grab a callable of the function to compute attribute value
            all_context_vars = {**globals(), **locals()}
            exec(att_txt, all_context_vars)
            compute_att_callables.append(all_context_vars[self.program_name])

        log.info('Attribute')
        for txt in att_txts:
            log.info(txt)
        log.info('Done')

        return compute_att_callables, att_txts

    async def _a_get_natural_language_observations(self,
                                                   c,
                                                   prompt,
                                                   compute_att_callable,
                                                   player_int_selector=False):
        observations = []
        to_be_prompteds = []
        for x in c:
            effects = self._get_natural_language_effects(x)

            if len(effects) == 0:
                continue

            input_target_obj_list = self.objects_selector(x.input_state)
            if player_int_selector:
                input_target_int_list = self.player_interactions_selector(
                    x.input_state.get_obj_interactions())
            else:
                input_target_int_list = self.interactions_selector(
                    x.input_state.get_obj_interactions())

            # Construct prompt
            to_be_prompteds.append(
                prompt.format(input=self._prep_interpret_input(
                    compute_att_callable, input_target_obj_list,
                    input_target_int_list),
                              effects=list_to_bullets(effects)))

        if len(to_be_prompteds) == 0:
            return []

        outputs = await self.llm.aprompt(to_be_prompteds,
                                         temperature=0,
                                         seed=self.config.seed)
        observations = sum([parse_listed_output(output) for output in outputs],
                           [])

        res = list(dict.fromkeys(observations))  # remove duplicates
        res.sort(
        )  # order don't matter -- trying to avoid unnecessary LLM calls
        return res

    def _get_natural_language_effects(self, x):
        if x not in self.cache_x:
            input_target_obj_list = self.objects_selector(x.input_state)
            output_target_obj_list = self.objects_selector(x.output_state)

            if len(input_target_obj_list) == 0 and len(
                    output_target_obj_list) == 0:
                self.cache_x[x] = []
                return self.cache_x[x]

            effects = []
            input_ids = [o.id for o in input_target_obj_list]
            for o in output_target_obj_list:
                if o.deleted == 1:
                    effects.append(f'The {o.str_w_id()} is deleted')
                elif o.id not in input_ids:
                    effects.append(
                        f'A new {o.obj_type} object is created at ' +
                        f'(x={o.x},y={o.y})')
                # no velocity

            self.cache_x[x] = effects
        return self.cache_x[x]

    def _prep_interpret_input(self,
                              compute_att_callable,
                              obj_list,
                              int_list=[]):
        txt1 = ''
        for obj in obj_list:
            txt1 += f'{obj.str_w_id()} with danger attribute = {compute_att_callable(obj)},\n'

        txt2 = ',\n'.join([xx.str_w_id() for xx in int_list
                           ]) if len(int_list) > 0 else 'No interactions'
        input_obs = txt1 + '\n' + txt2
        return input_obs
