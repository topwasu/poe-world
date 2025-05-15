from typing import Optional, List, Tuple, Union, Awaitable
import asyncio
import dill as pickle
import logging
from openai_hf_interface import create_llm
from omegaconf import DictConfig

from learners.synthesizer import Synthesizer
from learners.utils import *
from learners.models import *
from classes.helper import *
from programs import *

log = logging.getLogger('main')


class ObjModelLearner:
    """
    A class that learns object behavior models through observation of state
    transitions.
    Manages both creation and non-creation object behaviors using mixture of
    experts models.
    """
    def __init__(self, config: DictConfig, obj_type: str,
                 size_change_flag: bool, normal_modules: List[Synthesizer],
                 restart_modules: List[Synthesizer],
                 constraint_modules: List[Synthesizer],
                 pomdp_modules: List[Synthesizer]):
        """
        Initialize the object model learner with configuration and module specifications.
        
        Args:
            config: Configuration object containing learning parameters
            obj_type: Type of object to model
            normal_modules: Modules for normal state transitions
            restart_modules: Modules for handling game restarts
            constraint_modules: Modules for learning object constraints
            pomdp_modules: Modules for transitions conditioned on longer history
        """
        self.config = config
        self.obj_type = obj_type
        self.size_change_flag = size_change_flag

        # Create llm
        cache_mode = 'disk_to_memory' if self.config.use_memory else 'disk'
        self.llm = create_llm('gpt-4o-2024-08-06' if self.config.provider ==
                              'openai' else 'openai/gpt-4o-2024-08-06')
        self.llm.setup_cache(cache_mode, database_path=config.database_path)
        self.llm.set_default_kwargs({'timeout': 60})

        # Create synthesizers
        self.normal_synthesizers = [
            module(self.config, self.obj_type, self.llm)
            for module in normal_modules
        ]
        self.restart_synthesizers = [
            module(self.config, self.obj_type, self.llm)
            for module in restart_modules
        ]
        self.constraint_synthesizers = [
            module(self.config, self.obj_type, self.llm)
            for module in constraint_modules
        ]
        self.pomdp_synthesizers = [
            module(self.config, self.obj_type, self.llm)
            for module in pomdp_modules
        ]

        # Obj model learner config
        tmp = self.config.obj_model_learner
        self.save_freq = tmp.save_freq

        # Create static variables to be used
        self.objects_selector = ObjTypeObjSelector(self.obj_type)
        self.interactions_selector = ObjTypeInteractionSelector(self.obj_type)
        self.player_interactions_selector = ObjTypeInteractionSelector(
            'player')

        # Create modifiable variables
        self.processed_obs_count = 0
        self.transitions: List[StateTransitionTriplet] = []
        # hard coded values
        self.creation_keyword = 'create'
        self.batch_size = 10
        self.rng = np.random.default_rng(self.config.seed)

        # Create MoE models
        self.moe_non_creation = MoEObjModel(
            'non_creation',
            self.config,
            obj_type=self.obj_type,
            objects_selector=self.objects_selector.set_mode('non_creation'),
            size_change_flag=self.size_change_flag)
        self.moe_creation = MoEObjModel(
            'creation',
            self.config,
            obj_type=self.obj_type,
            objects_selector=self.objects_selector.set_mode('creation'),
            size_change_flag=self.size_change_flag)
        self.constraints = Constraints(self.obj_type,
                                       self.interactions_selector)

    def add_datapoint_and_infer_moe(self,
                                    x: StateTransitionTriplet) -> ObjTypeModel:
        # Adds a datapoint and triggers the inference process.
        self.add_datapoint(x)
        return self.infer_moe()

    def add_datapoint(self, x: StateTransitionTriplet) -> None:
        # Appends a single observation to the dataset.
        self.transitions.append(x)

    def return_obj_type_model(self) -> ObjTypeModel:
        # Returns object type model
        return ObjTypeModel(self.obj_type, self.moe_non_creation,
                            self.moe_creation, self.constraints)

    def return_constraints(self) -> Optional[Constraints]:
        if self.obj_type == 'player':
            return self.constraints
        return None

    def infer_moe(self) -> ObjTypeModel:
        """
        Main inference method that processes observations in batches and updates models.
        Handles checkpointing and weight fitting for both creation and non-creation behaviors.
        """
        # loop to sequentially infer posterior
        if self.load(None):
            return self.return_obj_type_model()

        # Load from recent checkpoints
        for chkpt in range(
                len(self.transitions) // self.save_freq * self.save_freq, 0,
                -self.save_freq):
            if self.load(chkpt):
                break

        log.info(f'At checkpoint {self.processed_obs_count}')
        while self.processed_obs_count < len(self.transitions):
            # Batch indices
            indices = np.arange(
                self.processed_obs_count,
                min(self.processed_obs_count + self.batch_size,
                    len(self.transitions)))

            # Run synthesizers
            log.info('--- Run fully observable MDP synthesizers ---')
            to_be_run_indices = self._grab_surprising_indices(indices)
            to_be_run = [
                self._a_infer_moe_at_transition(
                    self.transitions[:idx + 1],
                    with_constraint=(len(self.constraint_synthesizers) > 0))
                for idx in to_be_run_indices
            ]
            asyncio.run(await_gather(to_be_run))
            log.info(f'Current llm spending {self.llm.get_info()}')
            if len(to_be_run) > 0:
                self._update_moe(self.transitions[:indices[-1] + 1])
            else:
                log.info(
                    f'No new fully observable MDP rules for datapoints ({indices[0]} to {indices[-1]})'
                )
            log.info('--- Done running fully observable MDP synthesizers ---')

            # Run synthesizers with full history
            log.info('--- Run POMDP synthesizers ---')
            to_be_run_indices = self._grab_surprising_indices(indices)
            to_be_run = [
                self._a_infer_moe_at_transition(self.transitions[:idx + 1],
                                                with_constraint=False,
                                                with_full_history=True)
                for idx in to_be_run_indices
            ]
            asyncio.run(await_gather(to_be_run))
            log.info(f'Current llm spending {self.llm.get_info()}')
            if len(to_be_run) > 0:
                self._update_moe(self.transitions[:indices[-1] + 1])
            else:
                log.info(
                    f'No new POMDP rules for datapoints ({indices[0]} to {indices[-1]})'
                )
            log.info('--- Done running POMDP synthesizers ---')

            # Update checkpoint
            self.processed_obs_count = indices[-1] + 1

            log.info(f'Current llm spending {self.llm.get_info()}')

            if self.processed_obs_count % self.save_freq == 0:
                os.makedirs(
                    f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/{self.obj_type}',
                    exist_ok=True)
                self.save()

        self._update_moe(self.transitions, is_final=True)

        return self.return_obj_type_model()

    def slow_infer_moe(self) -> MoEObjModel:
        """
        Slower but more thorough inference method that processes observations sequentially.
        """
        log.info(f'At checkpoint {self.processed_obs_count}')
        to_be_run = []
        indices = np.arange(self.processed_obs_count, len(self.transitions))

        # Run synthesizers
        log.info('--- Run fully observable and POMDP synthesizers together ---')
        to_be_run_indices = self._grab_surprising_indices(indices, verbose=False)
        to_be_run = [
            self._a_infer_moe_at_transition(self.transitions[:idx + 1],
                                            with_constraint=False)
            for idx in to_be_run_indices
        ]
        to_be_run = to_be_run + [
            self._a_infer_moe_at_transition(self.transitions[:idx + 1],
                                            with_constraint=False,
                                            with_full_history=True)
            for idx in to_be_run_indices
        ]
        asyncio.run(await_gather(to_be_run))
        log.info(f'Current llm spending {self.llm.get_info()}')
        if len(to_be_run) > 0:
            self._update_moe(self.transitions, fast_fitting=False)
        else:
            log.info(
                f'No new rules for datapoints ({indices[0]} to {indices[-1]})'
            )
        log.info('--- Done running fully observable and POMDP synthesizers together ---')

        # Update checkpoint
        self.processed_obs_count = indices[-1] + 1

        os.makedirs(
            f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/{self.obj_type}',
            exist_ok=True)
        self.save()

        return self.return_obj_type_model()

    def fast_infer_moe(self) -> MoEObjModel:
        """
        Quick inference method that synthesizes functions for unexplained observations.
        """
        log.info(f'At checkpoint {self.processed_obs_count}')
        to_be_run = []
        indices = np.arange(self.processed_obs_count, len(self.transitions))

        # Run synthesizers
        log.info('--- Run fully observable MDP synthesizers ---')
        to_be_run_indices = self._grab_surprising_indices(indices, verbose=False)
        to_be_run = [
            self._a_infer_moe_at_transition(self.transitions[idx:idx + 1],
                                            with_constraint=False)
            for idx in to_be_run_indices
        ]
        asyncio.run(await_gather(to_be_run))
        log.info(f'Current llm spending {self.llm.get_info()}')
        if len(to_be_run) > 0:
            self._update_moe(self.transitions, fast_fitting=True)
        else:
            log.info(
                f'No new fully observable MDP rules for datapoints ({indices[0]} to {indices[-1]})'
            )
        log.info('--- Done running fully observable MDP synthesizers ---')

        # Update checkpoint
        self.processed_obs_count = indices[-1] + 1

        return self.return_obj_type_model()

    def _update_moe(self,
                    transitions: list[StateTransitionTriplet],
                    is_final: Optional[bool] = False,
                    fast_fitting: Optional[bool] = False):
        non_creation_c, creation_c = self._separate_creation_in_observation(
            transitions)

        if fast_fitting:
            log.info('Non creation weight fitting...')
            self.moe_non_creation.fit_only_new_weights(non_creation_c)
            log.info('Creation weight fitting...')
            self.moe_creation.fit_only_new_weights(creation_c)
        else:
            log.info('Non creation weight fitting...')
            self.moe_non_creation.fit_weights(non_creation_c)
            log.info('Creation weight fitting...')
            self.moe_creation.fit_weights(creation_c)

        # This could be slow
        # self.moe_non_creation.prune_programs_with_c(transitions)
        # self.moe_creation.prune_programs_with_c(transitions)
        self.moe_non_creation.prune_programs()
        self.moe_creation.prune_programs()
            

        if is_final:
            self.constraints.prune_programs(transitions)
            self.save(final=True)

    def _grab_surprising_indices(self, indices: List[int], verbose=True) -> List[int]:
        """
        Identify indices of observations that are not well explained by current models.
        """

        ret_indices = []
        for idx in indices:
            if not self._explain_well(idx):
                x = self.transitions[idx]
                if verbose:
                    log.info(f'Not explaining obs at {idx} well')
                    log.info(f'Inference at time {idx}')
                    log.info(
                        f"Trying to account for\nInput:\n{self.objects_selector(x.input_state)}\n"
                        +
                        f"Interactions:\n{self.player_interactions_selector(x.input_state.get_obj_interactions())}\n"
                        + f"Action:\n{x.event}\n" +
                        f"Output:\n{self.objects_selector(x.output_state)}")
                ret_indices.append(idx)
            else:
                if verbose:
                    log.info(f'No inference at time {idx}!')
        return ret_indices

    async def _a_infer_moe_at_transition(
            self,
            c: list[StateTransitionTriplet],
            with_constraint=True,
            with_full_history=False) -> Awaitable[None]:
        """
        Asynchronously infer new rules from observations using appropriate synthesizers.
        
        Args:
            c: List of state transition observations
            with_constraint: Whether to include constraint synthesis
            with_full_history: Whether to use the full history
        """
        if with_full_history and with_constraint:
            raise Exception(
                'Cannot have both with_full_history and with_constraint')
        if with_constraint and len(self.constraint_synthesizers) == 0:
            raise Exception(
                'Cannot have with_constraint if no constraint synthesizers are given'
            )
        log.debug('Running one iteration of synthesis')

        x = c[-1]

        # Get all async functions
        if x.output_game_state == GameState.RESTART:
            args = [
                synthesizer.a_synthesize(c)
                for synthesizer in self.restart_synthesizers
            ]
        else:
            if with_full_history:
                args = [
                    synthesizer.a_synthesize(c)
                    for synthesizer in self.pomdp_synthesizers
                ]
            else:
                args = [
                    synthesizer.a_synthesize(c)
                    for synthesizer in self.normal_synthesizers
                ]
                if with_constraint:
                    args = args + [
                        synthesizer.a_synthesize(c)
                        for synthesizer in self.constraint_synthesizers
                    ]

        # Call async functions
        new_rules = await asyncio.gather(*args)

        if x.output_game_state != GameState.RESTART and with_constraint:
            self.constraints.extend_rules(new_rules[-1], c[-1])
            new_rules = new_rules[:-1]

        new_rules = sum(new_rules, [])

        if x.output_game_state != GameState.RESTART and with_full_history:
            if len(new_rules) > 0:
                new_rules, new_context_lengths = zip(*new_rules)
                new_rules = list(new_rules)
                new_context_lengths = list(new_context_lengths)
            else:
                new_rules, new_context_lengths = [], []
        else:
            new_context_lengths = [-1] * len(new_rules)

        new_non_creation_rules, new_non_creation_context_lengths, new_creation_rules, new_creation_context_lengths = self._separate_creation_rules(
            new_rules, new_context_lengths)
        self.moe_non_creation.extend_rules(
            new_non_creation_rules,
            c,
            context_lengths=new_non_creation_context_lengths)
        self.moe_creation.extend_rules(
            new_creation_rules,
            c,
            context_lengths=new_creation_context_lengths)

        log.debug(f'Done')

    def _separate_creation_rules(
        self,
        rules: List[str],
        context_lengths: Optional[List[int]] = None
    ) -> Tuple[List[str], List[str]]:
        """Split rules into creation and non-creation related rules."""
        if context_lengths is None:
            return [
                rule for rule in rules if self.creation_keyword not in rule
            ], [rule for rule in rules if self.creation_keyword in rule]
        else:
            return [rule for rule in rules if self.creation_keyword not in rule], \
                [context_length for rule, context_length in zip(rules, context_lengths) if self.creation_keyword not in rule], \
                [rule for rule in rules if self.creation_keyword in rule], \
                [context_length for rule, context_length in zip(rules, context_lengths) if self.creation_keyword in rule], \

    def _separate_creation_in_observation(
        self, c: List[StateTransitionTriplet]
    ) -> Tuple[List[StateTransitionTriplet], List[StateTransitionTriplet]]:
        # Segregates creation events from non-creation events in observations.
        non_creation_c, creation_c = [], []
        for x in c:
            _, _, leftover_list2 = match_two_obj_lists(x.input_state,
                                                       x.output_state)
            non_creation_x = StateTransitionTriplet(
                x.input_state.deepcopy(), x.event,
                ObjList([
                    x.output_state[idx] for idx in range(len(x.output_state))
                    if idx not in leftover_list2
                ]))
            creation_x = StateTransitionTriplet(
                x.input_state.deepcopy(),
                x.event,
                ObjList([
                    x.output_state[idx] for idx in range(len(x.output_state))
                    if idx in leftover_list2
                ]),
                add_ghost=False)
            non_creation_c.append(non_creation_x)
            creation_c.append(creation_x)
        return non_creation_c, creation_c

    def _explain_well(self,
                      idx: int,
                      num: Optional[bool] = False) -> Union[bool, float]:
        """
        Check if current models can explain given observations well.
        
        Args:
            c: Observations to check
            num: If True, return numerical score instead of boolean
            full_arr, If True, return an array of length len(c) (don't reduce to single value)
        """
        try:
            x = self.transitions[idx]
            non_creation_c, creation_c = self._separate_creation_in_observation(
                [x])
            non_creation_x = non_creation_c[0]
            creation_x = creation_c[0]

            memory = x.input_state.memory
            obj1 = self.moe_non_creation.evaluate_logprobs(
                non_creation_x.input_state,
                non_creation_x.event,
                non_creation_x.output_state,
                memory=memory,
                params=np.clip(self.moe_non_creation.params, 0,
                               10),  # Clip just in case
                use_torch=False,
                precompute_index=-1)

            obj2 = self.moe_creation.evaluate_logprobs(
                creation_x.input_state,
                creation_x.event,
                creation_x.output_state,
                memory=memory,
                params=np.clip(self.moe_creation.params, 0,
                               10),  # Clip just in case
                use_torch=False,
                precompute_index=-1)

            obj1, obj2 = -obj1 * 1000, -obj2 * 1000
        except:
            log.info(
                'Exception occured in explain_well -- assuming it does not explain well'
            )
            obj1 = 1000000000
            obj2 = obj1

        if num:
            return np.max([obj1, obj2], axis=0) / 1000
        return (np.max([obj1, obj2], axis=0) / 1000) <= 0.2

    def load(self, checkpoint: Optional[int]) -> bool:
        """
        Loads model state from a checkpoint file.
        
        Args:
            checkpoint: Checkpoint number to load, or None for final checkpoint
            
        Returns:
            bool: True if load successful, False otherwise
        """
        path = self._get_checkpoint_path(checkpoint)
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        if len(tuple(data)) == 8:
            self._load_8_tuple_format(data)
        elif len(tuple(data)) == 9:
            self._load_9_tuple_format(data)
        elif len(tuple(data)) == 11:
            self._load_11_tuple_format(data)
        else:
            raise NotImplementedError("Unknown checkpoint format")

        self.processed_obs_count = checkpoint if isinstance(
            checkpoint, int) else len(self.transitions)
        log.info(f'Loaded checkpoint {path}')
        return True

    def _get_checkpoint_path(self, checkpoint: Optional[int]) -> str:
        """Gets the file path for the checkpoint"""
        if self.config.checkpoint_folder is not None:
            checkpoint_folder = self.config.checkpoint_folder
        else:
            checkpoint_folder = f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}'

        checkpoint_str = 'final' if checkpoint is None else str(checkpoint)
        path = f'{checkpoint_folder}/{self.obj_type}/{checkpoint_str}.pickle'
        return path

    def _load_9_tuple_format(self, data: tuple) -> None:
        """Loads data in legacy 9-tuple format"""
        (self.moe_non_creation.rules, self.moe_non_creation.params,
         self.moe_non_creation.fitteds, self.moe_non_creation.precompute_dist,
         self.moe_creation.rules, self.moe_creation.params,
         self.moe_creation.fitteds, self.moe_creation.precompute_dist,
         self.constraints.rules) = data

        self.moe_non_creation._prep_callables()
        self.moe_creation._prep_callables()
        self.constraints.prepare_callables()

        # We did not have pomdp modules and thus were not saving context lengths before
        self.moe_non_creation.context_lengths = [-1] * len(
            self.moe_non_creation.rules)
        self.moe_creation.context_lengths = [-1] * len(self.moe_creation.rules)

    def _load_11_tuple_format(self, data: tuple) -> None:
        """Loads data in current 11-tuple format"""
        (self.moe_non_creation.rules, self.moe_non_creation.params,
         self.moe_non_creation.fitteds, self.moe_non_creation.context_lengths,
         self.moe_non_creation.precompute_dist, self.moe_creation.rules,
         self.moe_creation.params, self.moe_creation.fitteds,
         self.moe_creation.context_lengths, self.moe_creation.precompute_dist,
         self.constraints.rules) = data

        self.moe_non_creation._prep_callables()
        self.moe_creation._prep_callables()
        self.constraints.prepare_callables()

    def save(self, final: bool = False) -> None:
        # Saves current model state to disk.
        log.info(
            f'Saving checkpoint {self.processed_obs_count} final = {final}')
        os.makedirs(
            f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/{self.obj_type}',
            exist_ok=True)
        data = [
            self.moe_non_creation.rules, self.moe_non_creation.params,
            self.moe_non_creation.fitteds,
            self.moe_non_creation.context_lengths,
            self.moe_non_creation.precompute_dist, self.moe_creation.rules,
            self.moe_creation.params, self.moe_creation.fitteds,
            self.moe_creation.context_lengths,
            self.moe_creation.precompute_dist, self.constraints.rules
        ]
        # with open(f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/reasoner_{self.obj_type}_{self.processed_obs_count}.pickle', "wb") as f:
        #     pickle.dump((self.moe_non_creation, self.moe_creation), f)
        if final:
            with open(
                    f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/{self.obj_type}/final.pickle',
                    "wb") as f:
                pickle.dump(data, f)
        else:
            with open(
                    f'saved_checkpoints_{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/{self.obj_type}/{self.processed_obs_count}.pickle',
                    "wb") as f:
                pickle.dump(data, f)

    def display_rules(self, mode='non_creation'):
        if mode == 'constraints':
            for idx, rule in enumerate(self.constraints.rules):
                print(f'Constraint #{idx + 1} for obj_type {self.obj_type}')
                print(rule)
        else:
            target = self.moe_non_creation if mode == 'non_creation' else self.moe_creation
            for idx, (rule, param, context_length) in enumerate(
                    zip(target.rules, target.params, target.context_lengths)):
                display_idx = idx + 1 if mode == 'non_creation' else idx + 1 + len(self.moe_non_creation.rules)
                print(f'Expert #{display_idx} for obj_type {self.obj_type} with weight = {param:.2f}')
                print(rule)
                # log.info(
                #     f'Rule {idx}, param = {param}, context length = {context_length}'
                # )
                # log.info(rule)
                # if idx > 0 and idx % 50 == 0:
                #     breakpoint()
                    
    def count_lines(self, mode='non_creation'):
        total_lines = 0
        if mode == 'constraints':
            for idx, rule in enumerate(self.constraints.rules):
                total_lines += len(rule.strip('\n').split('\n'))
        else:
            target = self.moe_non_creation if mode == 'non_creation' else self.moe_creation
            for idx, (rule, param, context_length) in enumerate(
                    zip(target.rules, target.params, target.context_lengths)):
                total_lines += len(rule.strip('\n').split('\n'))
        return total_lines
