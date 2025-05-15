import logging
from collections import defaultdict
from typing import Dict
from omegaconf import DictConfig
from abc import ABC, abstractmethod

from learners.obj_model_learner import ObjModelLearner
from learners.synthesizer import (
    ActionSynthesizer, PassiveMovementSynthesizer,
    NoInteractPassiveMovementSynthesizer, SnappingSynthesizer,
    RestartSynthesizer, ConstraintsSynthesizer, PlayerInteractionSynthesizer,
    MultiTimestepActionSynthesizer, MultiTimestepStatusChangeSynthesizer,
    MultiTimestepMomentumSynthesizer, VelocitySynthesizer,
    MultiTimestepSizeChangeSynthesizer, MultiTimestepStatusChangeSizeModeSynthesizer,
    MultiTimestepStatusChangeVelocityModeSynthesizer,
    MultiTimestepVelocitySynthesizer, PassiveCreationSynthesizer)
from learners.utils import *
from learners.models import *
from learners.models import WorldModel, Model, MoEObjModel, Constraints
from classes.helper import *

log = logging.getLogger('main')


class WorldModelLearner(ABC):
    @abstractmethod
    def synthesize_world_model(self, c: list[StateTransitionTriplet], **kwargs) -> Model:
        raise NotImplementedError
    
    @abstractmethod
    def update_world_model(self, c: list[StateTransitionTriplet], **kwargs) -> Model:
        raise NotImplementedError


class PoEWorldLearner(WorldModelLearner):
    """
    A class that learns world models by composing multiple object-specific models.
    Manages the learning of object behaviors and their interactions in the environment.
    """
    def __init__(self, config: DictConfig):
        """Initialize world model learner with configuration settings."""
        self.config = config
        self.obj_model_learners: Dict[str, ObjModelLearner] = {}
        # self.saved_world_model = None # doesn't seem to be used anywhere?
        self.saved_obj_model_learners: Dict[str, ObjModelLearner] = {}
        self.all_obj_types: Optional[List[str]] = None

    def _all_obj_types_in_obs(
            self, observations: list[ObjList]) -> Tuple[list[str], list[bool]]:
        """
        Extract all unique object types from a list of observations.
        Also return a list of flags indicating whether each object type has size changes.
        Exclude score-related objects if specified in config.
        Return only a single object type if specified in config (default = return only player).
        """
        if self.config.world_model_learner.obj_type != 'all':
            disappearing_tarpit_or_not = (self.config.world_model_learner.
                                          obj_type == 'disappearingtarpit')
            if self.config.rope_mode:
                return [self.config.world_model_learner.obj_type,
                        'rope'], [disappearing_tarpit_or_not, False]
            else:
                return [self.config.world_model_learner.obj_type
                        ], [disappearing_tarpit_or_not]
        else:
            size_flag_dict = defaultdict(bool)
            obj_type_dict = {}
            for obs in observations:
                for obj in obs:
                    if self.config.world_model_learner.exclude_score_objects and \
                            obj.obj_type in ['playerscore', 'enemyscore', 'score', 'timer', 'lifecount', 'life']:
                        continue
                    obj_type_dict[obj.obj_type] = True

                    if obj.w_change > 0 or obj.h_change > 0:
                        size_flag_dict[obj.obj_type] = True
            obj_types = list(obj_type_dict)
            return obj_types, [
                size_flag_dict[obj_type] for obj_type in obj_types
            ]

    def _init_obj_model_learners(self, obj_types: List[str],
                                size_change_flags: List[bool]) -> None:
        """
        Initialize object model learners for each object type.
        Use different synthesizers for player objects
        Size change flags indicate whether the object type has size changes.
        """
        for obj_type, size_change_flag in zip(obj_types, size_change_flags):
            if obj_type != 'player':
                normal_modules = [
                    NoInteractPassiveMovementSynthesizer,
                    PassiveMovementSynthesizer,
                    VelocitySynthesizer,
                    # PlayerInteractionSynthesizer
                ]
                restart_modules = []
                constraint_modules = []
                pomdp_modules = [
                    # MultiTimestepStatusChangeSynthesizer,
                    MultiTimestepVelocitySynthesizer,
                    MultiTimestepStatusChangeVelocityModeSynthesizer,
                    MultiTimestepMomentumSynthesizer
                ]

                if size_change_flag:
                    pomdp_modules.extend([
                        MultiTimestepSizeChangeSynthesizer,
                        MultiTimestepStatusChangeSizeModeSynthesizer
                    ])

            else:
                normal_modules = [
                    NoInteractPassiveMovementSynthesizer,
                    ActionSynthesizer,
                    PassiveMovementSynthesizer,
                    VelocitySynthesizer,
                    SnappingSynthesizer,
                    # VelocityTrackingSynthesizer
                ]
                restart_modules = [RestartSynthesizer,
                                   PassiveCreationSynthesizer]
                constraint_modules = [ConstraintsSynthesizer]
                pomdp_modules = [
                    MultiTimestepActionSynthesizer,
                ]

            self.obj_model_learners[obj_type] = ObjModelLearner(
                self.config, obj_type, size_change_flag, normal_modules,
                restart_modules, constraint_modules, pomdp_modules)

    def synthesize_world_model(
        self,
        c: list[StateTransitionTriplet]
    ) -> WorldModel:
        """
        Build a complete world model by learning models for each object type.
        
        Args:
            observations: List of object states at each timestep
            actions: List of actions taken between states
            kwargs: Additional arguments including game states
            
        Returns:
            A composed world model combining all object models
        """
        # Step 1: Extract unique object types from observations (e.g., 'player',
        # 'ball', 'brick')
        observations = [x.input_state for x in c] + [c[-1].output_state]
        self.all_obj_types, self.all_size_change_flags = self._all_obj_types_in_obs(
            observations)
        log.info(f'obj types found in observations {self.all_obj_types}')
        log.info(
            f'corresponding size change flags {self.all_size_change_flags}')

        # Step 2: Create model learners for each object type with appropriate
        # synthesizers
        self._init_obj_model_learners(self.all_obj_types,
                                     self.all_size_change_flags)

        obj_type_models: List[ObjTypeModel] = []
        constraints = None
        # Step 3: Learn models for each object type
        for obj_type in self.all_obj_types:
            log.info(f'Getting ObjModel for obj_type "{obj_type}"...')
            obj_model_learner = self.obj_model_learners[obj_type]

            # Step 3a: Create state transition triplets from consecutive
            # observations
            for x in c:
                obj_model_learner.add_datapoint(x)

            # Step 3b: Infer models and constraints for this object type
            obj_type_model = obj_model_learner.infer_moe()
            obj_type_models.append(obj_type_model)
            log.info(f'[Done] Getting ObjModel for obj_type {obj_type}!\n')

            if obj_type == 'player':
                constraints = obj_model_learner.return_constraints()

        # Step 4: Compose all object type models into a single world model
        self.world_model = WorldModel(obj_type_models, constraints)
        return self.world_model

    def update_world_model(self, c, fast=False, player_only=False) -> WorldModel:
        """
        Update existing world model with new observations.
        
        Args:
            c: New observations to incorporate
            fast: Whether to use fast inference mode
            
        Returns:
            Updated composed world model
        """
        obj_type_models: List[ObjTypeModel] = []
        constraints = None
        for obj_type in self.all_obj_types:
            if obj_type != 'player' and player_only:
                obj_type_models.append(self.obj_model_learners[obj_type].return_obj_type_model())
                continue
            log.info(f'Updating ObjModel for obj_type "{obj_type}" (fast={fast})...')
            obj_model_learner = self.obj_model_learners[obj_type]
            for x in c:
                obj_model_learner.add_datapoint(x)
            if fast:
                obj_type_model = obj_model_learner.fast_infer_moe()
            else:
                obj_type_model = obj_model_learner.slow_infer_moe()
            obj_type_models.append(obj_type_model)
            if obj_type == 'player':
                constraints = obj_model_learner.return_constraints()

        self.world_model = WorldModel(obj_type_models, constraints)
        return self.world_model

    def save_snapshot(self):
        """Save current state of all object model learners."""
        for obj_type in self.all_obj_types:
            self.saved_obj_model_learners[obj_type] = (
                copy.deepcopy(
                    self.obj_model_learners[obj_type].moe_non_creation),
                copy.deepcopy(self.obj_model_learners[obj_type].moe_creation),
                copy.deepcopy(self.obj_model_learners[obj_type].transitions),
                copy.deepcopy(
                    self.obj_model_learners[obj_type].processed_obs_count))

    def load_snapshot(self):
        """
        Restore saved state of object model learners and reconstruct world model.
        Returns the reconstructed composed world model.
        """
        for obj_type in self.all_obj_types:
            self.obj_model_learners[
                obj_type].moe_non_creation = self.saved_obj_model_learners[
                    obj_type][0]
            self.obj_model_learners[
                obj_type].moe_creation = self.saved_obj_model_learners[
                    obj_type][1]
            self.obj_model_learners[
                obj_type].transitions = self.saved_obj_model_learners[
                    obj_type][2]
            self.obj_model_learners[
                obj_type].processed_obs_count = self.saved_obj_model_learners[
                    obj_type][3]

        obj_type_models: List[ObjTypeModel] = []
        constraints = None
        for obj_type in self.all_obj_types:
            obj_model_learner = self.obj_model_learners[obj_type]
            obj_type_model = obj_model_learner.return_obj_type_model()
            obj_type_models.append(obj_type_model)
            if obj_type == 'player':
                constraints = obj_model_learner.return_constraints()

        self.world_model = WorldModel(obj_type_models, constraints)
        return self.world_model
