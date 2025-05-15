import copy
import numpy as np
import logging
import torch
import random
import pygame
from enum import Enum
from typing import Optional, Sequence, Union, Tuple, List

from classes.utils import *
from classes.game_utils import *

log = logging.getLogger('main')


class Constants:
    # Global constants to be set
    game_dict = None
    MAX_ABS_VELOCITY = None
    HISTORY_LENGTH = None
    MEMORY_LENGTH = None
    MAX_ABS_SIZE_CHANGE = None
    ACTIONS = None

    @staticmethod
    def set_constants(game_dict, max_abs_velocity, history_length,
                      max_abs_size_change, actions):
        Constants.game_dict = game_dict
        Constants.MAX_ABS_VELOCITY = max_abs_velocity
        Constants.HISTORY_LENGTH = history_length
        Constants.MEMORY_LENGTH = history_length
        Constants.MAX_ABS_SIZE_CHANGE = max_abs_size_change
        Constants.ACTIONS = actions


def set_global_constants(env_name):
    if env_name == 'MontezumaRevenge':
        Constants.set_constants(
            montezuma_revenge_wh_dict,
            MONTEZUMA_REVENGE_MAX_ABS_VELOCITY,
            MONTEZUMA_REVENGE_HISTORY_LENGTH,
            MONTEZUMA_REVENGE_MAX_ABS_SIZE_CHANGE,
            MONTEZUMA_REVENGE_ACTIONS
        )
    elif env_name == 'MontezumaRevengeAlt':
        Constants.set_constants(
            montezuma_revenge_wh_dict,
            MONTEZUMA_REVENGE_MAX_ABS_VELOCITY,
            MONTEZUMA_REVENGE_HISTORY_LENGTH,
            MONTEZUMA_REVENGE_MAX_ABS_SIZE_CHANGE,
            MONTEZUMA_REVENGE_ACTIONS
        )
    elif env_name == 'Pong':
        Constants.set_constants(
            pong_wh_dict, 
            PONG_MAX_ABS_VELOCITY,
            PONG_HISTORY_LENGTH,
            PONG_MAX_ABS_SIZE_CHANGE,
            PONG_ACTIONS
        )
    elif env_name == 'PongAlt':
        Constants.set_constants(
            pong_wh_dict, 
            PONG_MAX_ABS_VELOCITY,
            PONG_HISTORY_LENGTH,
            PONG_MAX_ABS_SIZE_CHANGE,
            PONG_ACTIONS
        )
    else:
        raise NotImplementedError


class GameState(str, Enum):
    DEAD = 'DIE'
    FROZEN = 'FREEZE'
    RESTART = 'RESTART'
    NORMAL = 'NORMAL'
    GAMEOVER = 'GAMEOVER'


# define a new exception class for when you die
class Died(Exception):
    pass


class StateTransitionTriplet:
    def __init__(self,
                 input_state: "ObjListWithMemory",
                 event: object,
                 output_state: "ObjListWithMemory",
                 input_game_state: Optional[GameState] = None,
                 output_game_state: Optional[GameState] = None,
                 add_ghost=True):
        """
        Represents a state transition with input state, event, output state, and optional game states.
        """
        self.input_state = input_state
        self.event = event
        self.output_state = output_state
        self.input_game_state = input_game_state
        self.output_game_state = output_game_state
        self.add_ghost = add_ghost

        # Set future ("new") positions and velocities for non-player objects
        for obj1 in input_state.objs:
            if obj1.obj_type == 'player':
                continue
            for obj2 in output_state.objs:
                if obj1.id == obj2.id:
                    obj1.set_new_pos_and_velocity(obj2.prev_x, obj2.prev_y,
                                                  obj2.velocity_x,
                                                  obj2.velocity_y)
                    break

        # add 'ghost' objects to output state
        if self.add_ghost:
            output_state_ids = [obj.id for obj in output_state.objs]
            new_output_state = output_state.deepcopy()
            for obj in input_state.objs:
                if obj.id not in output_state_ids:
                    new_obj = obj.copy()
                    new_obj.deleted = 1
                    new_obj.history['deleted'] = new_obj.history['deleted'][
                        -Constants.HISTORY_LENGTH:] + [1]
                    new_output_state.objs.append(new_obj)
            self.output_state = new_output_state

    def deepcopy(self) -> "StateTransitionTriplet":
        """
        Returns a deep copy of the current StateTransitionTriplet.
        """
        return StateTransitionTriplet(self.input_state.deepcopy(),
                                      copy.deepcopy(self.event),
                                      self.output_state.deepcopy(),
                                      self.input_game_state,
                                      self.output_game_state, False)

    def __repr__(self):
        return f'Input ObjList:\n{repr(self.input_state)}' + '\n' + f'Action: {repr(self.event)}' + '\n' + f'Output ObjList: {repr(self.output_state)}'

    # string with interactions
    def str_w_ints(self, player=False, player_int=False):
        return f'Input ObjList:\n{self.input_state.str_w_ints(player, player_int)}' + '\n' + f'Action: {repr(self.event)}' + '\n' + f'Output ObjList: {self.output_state.str_w_ints(player, player_int)}'


class StateMemory:
    def __init__(self, max_length):
        self.memory = []
        self.max_length = max_length
        
    def reset(self):
        self.memory = []
        
    def copy(self):
        new_obj = StateMemory(self.max_length)
        new_obj.memory = self.memory.copy()
        return new_obj
        
    def add_obj_list_and_action(self, obj_list, action):
        """
        Add a new obj list and action to the memory 
        If objects are deleted in the new obj list, add a ghost object
        """
        
        # Don't add objlistwithmemory in memory -- ram might blows up
        if isinstance(obj_list, ObjListWithMemory):
            obj_list = obj_list.convert_to_obj_list()
        
        if len(self.memory) > 0:
            prev_obj_list = self.memory[-1][0]
            # add 'ghost' objects to output state
            output_state_ids = [obj.id for obj in obj_list.objs]
            new_output_state = obj_list.deepcopy()
            for obj in prev_obj_list.objs:
                if obj.id not in output_state_ids and obj.deleted == 0:
                    new_obj = obj.copy()
                    new_obj.deleted = 1
                    new_obj.history['deleted'] = new_obj.history['deleted'][
                        -Constants.HISTORY_LENGTH:] + [1]
                    new_output_state.objs.append(new_obj)
            obj_list = new_output_state

        # This creates a new copy
        self.memory = self.memory + [(obj_list, action)]
        
        # Truncate
        self.memory = self.memory[-self.max_length:]
        
    def __getitem__(self, idx):
        return self.memory[idx]
    
    def __len__(self):
        return len(self.memory)
        


class RandomValues:
    def __init__(self,
                 values: Sequence[float],
                 logscores: Optional[np.ndarray] = None,
                 use_torch: bool = False) -> None:
        """
        Holds numerical values and logscores, supports sampling and logprob evaluation.
        """
        self.values = np.array(values)
        if logscores is None:
            self.logscores = np.zeros_like(self.values)
        else:
            self.logscores = logscores
        self.logprobs = None
        self.max_prob_value = None
        self.use_torch = use_torch
        self.logsumexp = torch.logsumexp if use_torch else logsumexp
        self.exp_normalize = torch_exp_normalize if use_torch else exp_normalize
        self.temp = 1

    def fit(self, obs):
        raise NotImplementedError

    def evaluate_logprobs(self, value: float, temp: float = 1) -> float:
        if self.logprobs is None or self.temp != temp:
            self.logprobs = self.logscores / temp - self.logsumexp(
                self.logscores / temp, -1)
            self.temp = temp
        if value in self.values:
            return self.logprobs[np.where(self.values == value)[0][0]]
        return LOG_IMPOSSIBLE_VALUE

    def sample(self,
               temp: float = 1,
               rng: Optional[random.Random] = None) -> float:
        if self.use_torch:
            raise NotImplementedError
        else:
            if rng is not None:
                return rng.choice(self.values,
                                  p=self.exp_normalize(self.logscores / temp))
            else:
                return np.random.choice(self.values,
                                        p=self.exp_normalize(self.logscores /
                                                             temp))

    def get_max_prob_value(self):
        if self.max_prob_value is None:
            self.max_prob_value = self.values[np.argmax(self.logscores)]
        return self.max_prob_value

    def get_value(self):
        return self

    def __repr__(self) -> str:
        if self.logprobs is None:
            self.logprobs = self.logscores - self.logsumexp(self.logscores, -1)
        txt = ' '.join(
            [f"({x}, {y:.2f})" for x, y in zip(self.values, self.logprobs)])
        return f'RandomValues | Values and LogProbs [{txt}]'

    def __getitem__(self, idx):
        if self.logprobs is None:
            self.logprobs = self.logscores - self.logsumexp(self.logscores, -1)
        return self.values[idx], self.logprobs[idx]

    def __add__(self, other):
        if isinstance(other, RandomValues):
            raise NotImplementedError
        else:
            return RandomValues(self.values + other,
                                logscores=self.logscores,
                                use_torch=self.use_torch)

    def __radd__(self, other):
        if isinstance(other, RandomValues):
            raise NotImplementedError
        else:
            return RandomValues(self.values + other,
                                logscores=self.logscores,
                                use_torch=self.use_torch)

    def __sub__(self, other):
        if isinstance(other, RandomValues):
            raise NotImplementedError
        else:
            return RandomValues(self.values - other,
                                logscores=self.logscores,
                                use_torch=self.use_torch)

    def __rsub__(self, other):
        if isinstance(other, RandomValues):
            raise NotImplementedError
        else:
            return RandomValues(self.values - other,
                                logscores=self.logscores,
                                use_torch=self.use_torch)


class SeqValues:
    def __init__(self, sequence):
        self.sequence = sequence

    def __repr__(self):
        return f'SeqValues | Sequence {self.sequence}'

    def __getitem__(self, idx):
        return self.sequence[idx]

    def __len__(self):
        return len(self.sequence)


class Interaction:
    def __init__(self, obj1: "Obj", obj2: "Obj") -> None:
        """
        Represents an interaction between two objects.
        """
        self.obj1 = obj1
        self.obj2 = obj2

    def flip(self):
        return Interaction(self.obj2, self.obj1)

    def equals(self, input):
        if self.obj1.equals(input.obj1) and self.obj2.equals(input.obj2):
            return True
        elif self.obj1.equals(input.obj2) and self.obj2.equals(input.obj1):
            return True
        return False

    def str_w_id(self):
        return f"Interaction -- {self.obj1.str_w_id()} is touching {self.obj2.str_w_id()}"

    def str_wo_id(self):
        return f"Interaction -- {self.obj1.str_wo_id()} is touching {self.obj2.str_wo_id()}"

    def __repr__(self):
        return f"Interaction -- {repr(self.obj1)} is touching {repr(self.obj2)}"


class Obj:
    def __init__(self,
                 game_object: Optional[object] = None,
                 id: int = -1,
                 deleted: int = 0,
                 obj_type: Optional[str] = None,
                 x: Optional[int] = None,
                 y: Optional[int] = None) -> None:
        """
        Represents an in-game object with type, position, velocity, and optional ID.
        """
        if game_object is None and \
                obj_type is not None and \
                x is not None and \
                y is not None:
            self.obj_type = obj_type
            self.w, self.h = Constants.game_dict[obj_type]

            self.prev_x = x
            self.prev_y = y
            self.velocity_x, self.velocity_y = 0, 0
            self.id = -1
            self.deleted = 0

            self.prev_velocity_x = 0
            self.prev_velocity_y = 0
        else:
            self.obj_type = game_object.category.lower()
            self.w = game_object.w
            self.h = game_object.h
            self.prev_x, self.prev_y = game_object.prev_xy if game_object.prev_xy != (
                0, 0) else (game_object.x, game_object.y)
            self.velocity_x = game_object.dx if game_object.prev_xy != (
                0, 0) else 0
            self.velocity_y = game_object.dy if game_object.prev_xy != (
                0, 0) else 0
            self.id = id
            self.deleted = deleted

            self.prev_velocity_x = None
            self.prev_velocity_y = None

        self.w_change = 0
        self.h_change = 0

        self.history = {
            'velocity_x': [],
            'velocity_y': [],
            'deleted': [1],
            'touch_below': [],
            'w_change': [],
            'h_change': []
        }

        self.new_prev_x = None
        self.new_prev_y = None
        self.new_velocity_x = None
        self.new_velocity_y = None

        self.new_center_x = None
        self.new_center_y = None
        self.new_left_side = None
        self.new_right_side = None
        self.new_top_side = None
        self.new_bottom_side = None

    def set_new_pos_and_velocity(self, prev_x, prev_y, velocity_x, velocity_y):
        self.new_prev_x = prev_x
        self.new_prev_y = prev_y
        self.new_velocity_x = velocity_x
        self.new_velocity_y = velocity_y

        new_x = prev_x + velocity_x
        new_y = prev_y + velocity_y

        self.new_center_x = new_x + self.w // 2
        self.new_center_y = new_y + self.h // 2
        self.new_left_side = new_x
        self.new_right_side = new_x + self.w
        self.new_top_side = new_y
        self.new_bottom_side = new_y + self.h

    @property
    def x(self):
        return self.prev_x + self.velocity_x

    @property
    def y(self):
        return self.prev_y + self.velocity_y

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def prev_xy(self):
        return (self.prev_x, self.prev_y)

    @property
    def center_x(self):
        return self.x + self.w // 2

    @property
    def center_y(self):
        return self.y + self.h // 2

    @property
    def left_side(self):
        return self.x

    @property
    def right_side(self):
        return self.x + self.w

    @property
    def top_side(self):
        return self.y

    @property
    def bottom_side(self):
        return self.y + self.h

    @center_x.setter
    def center_x(self, value):
        x = value - self.w // 2
        velocity_x = x - self.prev_x - self.prev_velocity_x  # bc of step
        self.velocity_x = velocity_x

    @center_y.setter
    def center_y(self, value):
        y = value - self.h // 2
        velocity_y = y - self.prev_y - self.prev_velocity_y  # bc of step
        self.velocity_y = velocity_y

    @left_side.setter
    def left_side(self, value):
        x = value
        velocity_x = x - self.prev_x - self.prev_velocity_x  # bc of step
        self.velocity_x = velocity_x

    @right_side.setter
    def right_side(self, value):
        x = value - self.w
        velocity_x = x - self.prev_x - self.prev_velocity_x  # bc of step
        self.velocity_x = velocity_x

    @top_side.setter
    def top_side(self, value):
        y = value
        velocity_y = y - self.prev_y - self.prev_velocity_y  # bc of step
        self.velocity_y = velocity_y

    @bottom_side.setter
    def bottom_side(self, value):
        y = value - self.h
        velocity_y = y - self.prev_y - self.prev_velocity_y  # bc of step
        self.velocity_y = velocity_y

    @property
    def center(self):
        return (self.center_x, self.center_y)

    @property
    def upper_y(self):
        return self.y

    @property
    def lower_y(self):
        return self.y + self.h

    @property
    def left_x(self):
        return self.x

    @property
    def right_x(self):
        return self.x + self.w

    @property
    def falling_time(self):  # TODO: can be optimized
        if 'touch_below' not in self.history or len(
                self.history['touch_below']) != len(
                    self.history['velocity_y']):
            return 0
        falling_ct = 0
        for x, y in zip(self.history['velocity_y'][::-1][1:],
                        self.history['touch_below'][::-1][1:]):
            if x <= 0 or y:
                break
            falling_ct += 1
        return falling_ct + 1 if falling_ct > 0 else 0

    def closest_object(self, objects):
        return min([(self.distance_to(o), o) for o in objects])

    def distance_to(self, other):
        return np.sqrt((self.center_x - other.center_x)**2 +
                       (self.center_y - other.center_y)**2)

    def old_touches(self, other, shrink_top=0, shrink_bottom=0, shrink_side=0):
        if self.obj_type == other.obj_type:
            return False
        # TODO: Fix this, don't stop at center
        return pygame.Rect(min(self.x - 1 + shrink_side, self.center_x),
                           min(self.y - 1 + shrink_top, self.center_y),
                           max(1, self.w + 2 - shrink_side),
                           max(1, self.h + 2 - shrink_bottom)).colliderect(
                               pygame.Rect(other.x, other.y, other.w, other.h))
                           
    def overlaps(self, other):
        return pygame.Rect(self.x - 1, self.y, self.w + 2, self.h).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))

    def touches(self, other, touch_side=-1, touch_percent=0):
        if self.obj_type == other.obj_type:
            return False
        if touch_side == -1:
            return pygame.Rect(self.x - 1, self.y - 1, self.w + 2,
                               self.h + 2).colliderect(
                                   pygame.Rect(other.x, other.y, other.w,
                                               other.h))
        elif touch_side == 0:  # LEFT
            return pygame.Rect(self.x - 1,
                               self.y,
                               self.w // 2,
                               max(int(self.h * (1-touch_percent)), 1)).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))\
                and pygame.Rect(self.x - 1,
                                self.y + self.h - max(int(self.h * (1-touch_percent)), 1),
                                self.w // 2,
                                max(int(self.h * (1-touch_percent)), 1)).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))
        elif touch_side == 1:  # RIGHT
            return pygame.Rect(self.x + self.w - self.w // 2 + 1,
                               self.y,
                               self.w // 2,
                               max(int(self.h * (1-touch_percent)), 1)).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))\
                and pygame.Rect(self.x + self.w - self.w // 2 + 1,
                                self.y + self.h - max(int(self.h * (1-touch_percent)), 1),
                                self.w // 2,
                                max(int(self.h * (1-touch_percent)), 1)).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))
        elif touch_side == 2:  # TOP
            return pygame.Rect(self.x,
                               self.y - 1,
                               max(int(self.w * (1-touch_percent)), 1),
                               self.h // 2).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))\
                and pygame.Rect(self.x + self.w - max(int(self.w * (1-touch_percent)), 1),
                                self.y - 1,
                                max(int(self.w * (1-touch_percent)), 1),
                                self.h // 2).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))
        elif touch_side == 3:  # BOTTOM
            return pygame.Rect(self.x,
                               self.y + self.h - self.h // 2 + 1,
                               max(int(self.w * (1-touch_percent)), 1),
                               self.h // 2).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))\
                and pygame.Rect(self.x + self.w - max(int(self.w * (1-touch_percent)), 1),
                                self.y + self.h - self.h // 2 + 1,
                                max(int(self.w * (1-touch_percent)), 1),
                                self.h // 2).colliderect(pygame.Rect(other.x, other.y, other.w, other.h))

    def deepcopy(self):
        raise NotImplementedError
        # return Obj(self.game_object)

    def str_w_id(self):
        return f"{self.obj_type} object (id = {self.id})"

    def str_wo_id(self):
        return f"{self.obj_type} object"

    def pre_step(self) -> None:
        self.prev_velocity_x = self.velocity_x
        self.prev_velocity_y = self.velocity_y

    def step(self) -> None:
        """
        Updates internal state after each time step (position, velocity, etc.).
        """
        self.prev_x = self.prev_x + self.prev_velocity_x
        self.prev_y = self.prev_y + self.prev_velocity_y

        # TODO: Think through this -- do not assume it's going at the same velocity, otherwise objects might be thought to interact before they actually do)
        # self.velocity_x = 0
        # self.velocity_y = 0

    def size_propagate(self) -> None:
        if isinstance(self.w_change, RandomValues) or isinstance(
                self.h_change, RandomValues) or isinstance(
                    self.x, RandomValues) or isinstance(self.y, RandomValues):
            # Here we decide to not propagate size changes when it's still random variables
            # Because we choose to keep center_x center_y fixed
            # So if they are random variables, self.prev_x self.prev_y will be random variables in a complicated way
            # Then self.x self.y will be each be a combination of two random variables ==> headache
            raise Exception('w and h cannot be RandomValues in size_propagate')

        old_center_x = self.center_x
        old_center_y = self.center_y

        if self.w_change < 0:
            self.w_change = max(self.w_change, -self.w)
        if self.h_change < 0:
            self.h_change = max(self.h_change, -self.h)
        self.w += self.w_change
        self.h += self.h_change

        # keep center_x and center_y still
        self.prev_x -= self.center_x - old_center_x
        self.prev_y -= self.center_y - old_center_y

    def copy(self):
        new_instance = copy.copy(self)
        new_instance.history = copy.copy(
            new_instance.history)  # Copy history bc dict is a mutable object
        return new_instance

    def __repr__(self):
        if self.obj_type.lower() != 'player':
            return f"{self.obj_type} object at (x={self.x}, y={self.y})"
        else:
            return f"{self.obj_type} object at (x={self.x}, y={self.y}, falling_time={self.falling_time}, velocity x={self.velocity_x}, y={self.velocity_y})"


class ObjList:
    def __init__(self, objs: Sequence[Obj], no_copy: bool = False) -> None:
        """
        Holds a list of Obj instances, with optional copying of objects.
        """
        self.objs = [
            obj.copy() if not no_copy else obj for obj in objs
            if obj is not None
        ]

    def __repr__(self):
        return ',\n'.join([str(obj) for obj in self.objs])

    def str_w_id(self):
        return ',\n'.join([obj.str_w_id() for obj in self.objs])

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        return self.objs[idx]

    # string with interactions
    def str_w_ints(self, player=False, player_int=False):
        objs = self.objs
        if player:
            objs = self.get_objs_by_obj_type('player')
        txt1 = ',\n'.join([str(obj)
                           for obj in objs]) if len(objs) > 0 else 'No objects'
        interactions = self.get_obj_interactions()
        if player_int:
            interactions = [
                xx for xx in interactions
                if xx.obj1.obj_type == 'player' or xx.obj2.obj_type == 'player'
            ]
        txt2 = ',\n'.join([repr(xx) for xx in interactions
                           ]) if len(interactions) > 0 else 'No interactions'
        input_obs = txt1 + ',\n' + txt2
        return input_obs

    def get_objs_by_obj_type(self, obj_type):
        return [obj for obj in self.objs if obj.obj_type == obj_type]

    def get_obj_by_id(self, id):
        return [obj for obj in self.objs if obj.id == id][0]

    def deepcopy(self) -> "ObjList":
        """
        Returns a deep copy of the current ObjList.
        """
        return ObjList(self.objs)

    def get_obj_interactions(self):
        res = []

        # Ordered pair
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                if self.objs[i].touches(self.objs[j]):
                    res.append(Interaction(self.objs[i], self.objs[j]))
        return res

    def get_player_interactions(self, player_first=False):
        interactions = self.get_obj_interactions()
        interactions = [
            xx for xx in interactions
            if xx.obj1.obj_type == 'player' or xx.obj2.obj_type == 'player'
        ]
        if player_first:
            interactions = [
                xx if xx.obj1.obj_type == 'player' else xx.obj2.flip()
                for xx in interactions
            ]
        return interactions

    def pre_step(self):
        for obj in self.objs:
            obj.pre_step()

    def step(self):
        for obj in self.objs:
            obj.step()

    def add_object(self, obj_type, x, y):
        new_obj = Obj(obj_type=obj_type, x=x, y=y)
        return new_obj

    def create_object(self, obj_type, x, y, delay_timesteps=-1):
        # Do not use this in normal situations -- only for LLM
        # It deletes everything leaving just the new object in the list
        # And it sets everything to randomvalues to avoid getting replaced with uniform dist
        if isinstance(x, RandomValues) or isinstance(y, RandomValues):
            raise Exception('x and y cannot be RandomValues in create_object')

        obj = self.add_object(obj_type, x, y)

        turn_all_atts_to_random_values(obj)

        if delay_timesteps > 0:
            obj.deleted = SeqValues([1] * (delay_timesteps - 1) + [0])

        # Add objects -- make sure it is sorted by prev_x and prev_y
        new_objs = []
        added = False
        for o in self.objs:
            if o.id == -1:
                if not added and (o.prev_x > obj.prev_x or
                                  (o.prev_x == obj.prev_x
                                   and o.prev_y >= obj.prev_y)):
                    new_objs.append(obj)
                    added = True

                if o.prev_x == obj.prev_x and o.prev_y == obj.prev_y:
                    # Don't put old obj in if exactly the same as the new object
                    continue
                new_objs.append(o)
        if not added:
            new_objs.append(obj)
            added = True

        return ObjList(new_objs, no_copy=True)

    def remove_deleted_objs(self):
        return ObjList([obj for obj in self.objs if obj.deleted == 0])
    
    def get_str_w_ints_w_touching(self, w_id=False):
        res = 'List of objects:\n'
        
        for obj in self.objs:
            if w_id:
                res += f"{obj.obj_type} object (id = {obj.id}) with x={obj.x}, y={obj.y}\n"
            else:
                res += f"{obj.obj_type} object with x={obj.x}, y={obj.y}, velocity_x={obj.velocity_x}, velocity_y={obj.velocity_y}\n"
        
        res += '\nList of object interactions:\n'
        
        # Ordered pair
        at_least_one_touching = False
        for i in range(len(self)):
            for j in range(len(self)):
                # skip if same object
                if i == j:
                    continue
                
                # skip if not touching
                if not self.objs[i].touches(self.objs[j]):
                    continue
                
                best_touching_side = -1
                best_touching_percent = 0
                for side in range(4):
                    for percent in np.arange(0.1, 1.1, 0.1):
                        if percent > best_touching_percent and self.objs[i].touches(self.objs[j], touch_side=side, touch_percent=percent):
                            best_touching_side = side
                            best_touching_percent = percent
                
                if w_id:
                    if best_touching_side == -1:
                        touch_side_txt = 'overlap'
                    else:
                        touch_side_txt = ['left', 'right', 'top', 'bottom'][best_touching_side]
                    res += f'{self.objs[i].obj_type} (id = {self.objs[i].id}) touches {self.objs[j].obj_type} (id = {self.objs[j].id}) with touch_side={touch_side_txt} and touch_percent={best_touching_percent}\n'
                else:
                    res += f'{self.objs[i].obj_type} touches {self.objs[j].obj_type} with touch_side={best_touching_side} and touch_percent={best_touching_percent}\n'
                at_least_one_touching = True
                
        if not at_least_one_touching:
            res += 'No touching objects\n'
            
        return res

class ObjSelector:
    def __init__(self, mode: str = 'non_creation') -> None:
        """
        Selects objects from a state transition under a given mode (creation or non_creation).
        """
        if mode not in ['non_creation', 'creation']:
            raise Exception(
                f"selected mode '{mode}' not in our options (non_creation, creation)"
            )
        self.mode = mode

    def set_mode(self, mode):
        return ObjSelector(mode)

    def __call__(self, next_state, prev_state=None):
        if self.mode == 'creation' and prev_state is None:
            raise Exception('prev_state cannot be None in mode=creation')

        if self.mode == 'non_creation':
            if prev_state is None:
                return next_state
            _, _, leftover_list2 = match_two_obj_lists(prev_state, next_state)
            obj_list = ObjList([
                next_state[idx] for idx in range(len(next_state))
                if idx not in leftover_list2
            ])
            return obj_list
        else:
            _, _, leftover_list2 = match_two_obj_lists(prev_state, next_state)

            # In creation mode, if id == -1, automatically add to leftover_list2
            for idx, o in enumerate(next_state):
                if o.id == -1 and idx not in leftover_list2:
                    leftover_list2.append(idx)

            obj_list = ObjList([next_state[idx] for idx in leftover_list2])
            return obj_list


class ObjTypeObjSelector(ObjSelector):
    def __init__(self, obj_type: str, mode: str = 'non_creation') -> None:
        """
        Selects objects of a specific type (obj_type) under the given mode.
        """
        if mode not in ['non_creation', 'creation']:
            raise Exception(
                f"selected mode '{mode}' not in our options (non_creation, creation)"
            )
        self.mode = mode
        self.obj_type = obj_type

    def set_mode(self, mode):
        return ObjTypeObjSelector(self.obj_type, mode)

    def __call__(self, next_state, prev_state=None):
        state = super().__call__(next_state, prev_state)
        res = ObjList(state.get_objs_by_obj_type(self.obj_type))
        return res


class ObjTypeInteractionSelector:
    def __init__(self, obj_type: str) -> None:
        """
        Filters a list of interactions to only those involving a specific object type.
        """
        self.obj_type = obj_type

    def __call__(self, interactions):
        return [
            ia for ia in interactions if ia.obj1.obj_type == self.obj_type
            or ia.obj2.obj_type == self.obj_type
        ]
        
        
class ObjListWithMemory(ObjList):
    def __init__(self, obj_list, memory):
        self.obj_list = obj_list
        self.objs = obj_list.objs
        self.memory = memory.copy()
        
    def deepcopy(self):
        return ObjListWithMemory(self.obj_list.deepcopy(), self.memory.copy())
    
    def convert_to_obj_list(self):
        return self.obj_list


def turn_all_atts_to_random_values(obj: Obj):
    if not isinstance(obj.velocity_x, RandomValues):
        obj.velocity_x = RandomValues([obj.velocity_x])
    if not isinstance(obj.velocity_y, RandomValues):
        obj.velocity_y = RandomValues([obj.velocity_y])
    if not isinstance(obj.deleted, RandomValues):
        obj.deleted = RandomValues([obj.deleted])
    # Don't turn w_change and h_change to RandomValues
    # fill_unset_values_with_uniform has a size flag so this wouldn't be a problem
    # if not isinstance(obj.w_change, RandomValues):
    #     obj.w_change = RandomValues([obj.w_change])
    # if not isinstance(obj.h_change, RandomValues):
    #     obj.h_change = RandomValues([obj.h_change])


def add_noise_to_random_values(random_values: RandomValues, min_range: int,
                               max_range: int,
                               log_noise_value: float) -> RandomValues:
    """
    Adds noise to RandomValues by expanding possible values and updating logscores.
    """
    all_possible_values = np.arange(min_range, max_range + 1)
    new_logscores = np.full(len(all_possible_values), log_noise_value)
    for val, logscore in zip(random_values.values, random_values.logscores):
        if val < min_range or val > max_range:
            continue
        new_logscores[val - min_range] = max(logscore, log_noise_value)
    return RandomValues(all_possible_values, logscores=new_logscores)


def combine_random_values(random_values_lst: Sequence[RandomValues],
                          weights: np.ndarray,
                          use_torch: bool) -> RandomValues:
    """
    Combines multiple RandomValues using weighted averaging of logscores.
    """
    if use_torch:
        return combine_random_values_torch(random_values_lst, weights)

    values = random_values_lst[0].values
    logscores_lst = []

    for i in range(len(random_values_lst)):
        # if i + 1 < len(random_values_lst):
        #     if len(random_values_lst[i].values) != len(random_values_lst[i + 1].values):
        #         raise Exception('the values in RandomValues have different length')
        #     if (random_values_lst[i].values != random_values_lst[i + 1].values).any():
        #         raise Exception('the values in RandomValues are different')

        logscores_lst.append(random_values_lst[i].logscores)

    logscores = np.stack(logscores_lst).T @ weights
    res = RandomValues(values, logscores=logscores)
    return res


def combine_random_values_torch(random_values_lst: Sequence[RandomValues],
                                weights: torch.Tensor) -> RandomValues:
    """
    Torch-based version of combine_random_values.
    """
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    values = random_values_lst[0].values

    logscores = torch.tensor(np.asarray([
        random_values_lst[i].logscores for i in range(len(random_values_lst))
    ],
                                        dtype=np.float32),
                             dtype=torch.float32,
                             requires_grad=False).to(device).T @ weights
    res = RandomValues(values, logscores=logscores, use_torch=True)
    return res


def instantiate_obj_list(obj_list_dist: "ObjList",
                         det: bool = False,
                         temp: float = 1,
                         rng: Optional[random.Random] = None) -> "ObjList":
    """
    Takes an ObjList distribution and returns a sampled or deterministic ObjList.
    """
    if obj_list_dist is None:
        return ObjList([])

    if rng is None:
        rng = np.random.default_rng(np.random.randint(1000))

    obj_list = obj_list_dist.deepcopy()
    max_id = -1
    for obj in obj_list:
        # using this hack to make the beams appear at the same time?
        rng_copy = copy.deepcopy(rng)

        max_id = max(max_id, obj.id)
        if isinstance(obj.velocity_x, RandomValues):
            if det:
                obj.velocity_x = obj.velocity_x.get_max_prob_value()
            else:
                obj.velocity_x = obj.velocity_x.sample(temp, rng_copy)
        if isinstance(obj.velocity_y, RandomValues):
            if det:
                obj.velocity_y = obj.velocity_y.get_max_prob_value()
            else:
                obj.velocity_y = obj.velocity_y.sample(temp, rng_copy)
        if isinstance(obj.deleted, RandomValues):
            if det:
                obj.deleted = obj.deleted.get_max_prob_value()
            else:
                obj.deleted = obj.deleted.sample(temp, rng_copy)

        # Deal with sizes too
        if isinstance(obj.w_change, RandomValues):
            if det:
                obj.w_change = obj.w_change.get_max_prob_value()
            else:
                obj.w_change = obj.w_change.sample(temp, rng_copy)
        if isinstance(obj.h_change, RandomValues):
            if det:
                obj.h_change = obj.h_change.get_max_prob_value()
            else:
                obj.h_change = obj.h_change.sample(temp, rng_copy)
        # can finally propagate size -- changing w, h, prev_x, prev_y
        obj.size_propagate()

    # give id to new objects
    for obj in obj_list:
        if obj.id == -1:
            obj.id = max_id + 1
            max_id += 1

    return obj_list.remove_deleted_objs()


def evaluate_logprobs_of_obj_list(obj_list_dist: "ObjList",
                                  obj_list: "ObjList",
                                  by_pos: bool = False,
                                  size_change_flag: bool = False) -> float:
    """
    Computes the log-probabilities of matching an ObjList distribution to actual objects.
    If by_pos = True, evaluate logprobs by position (x, y) instead of velocity (for object creation).
    If size_change_flag = True, additionally evaluate logprobs by size change (w_change, h_change) as well.
    """
    if obj_list_dist is None:
        return LOG_IMPOSSIBLE_VALUE

    good_pairs, leftover_list1, leftover_list2 = match_two_obj_lists(
        obj_list_dist, obj_list)

    logprobs = 0
    for idx1, idx2 in good_pairs:
        obj_dist = obj_list_dist[idx1]
        obj = obj_list[idx2]

        if obj.deleted != 1:
            if by_pos:
                if isinstance(obj_dist.x, RandomValues):
                    logprobs += obj_dist.x.evaluate_logprobs(obj.x)
                else:
                    logprobs += 0 if obj_dist.x == obj.x else LOG_IMPOSSIBLE_VALUE
                if isinstance(obj_dist.y, RandomValues):
                    logprobs += obj_dist.y.evaluate_logprobs(obj.y)
                else:
                    logprobs += 0 if obj_dist.y == obj.y else LOG_IMPOSSIBLE_VALUE
            else:
                # If size change, use velocity instead
                # This still ruins creation TODO
                if isinstance(obj_dist.velocity_x, RandomValues):
                    logprobs += obj_dist.velocity_x.evaluate_logprobs(
                        obj.velocity_x)
                else:
                    logprobs += 0 if obj_dist.velocity_x == obj.velocity_x else LOG_IMPOSSIBLE_VALUE
                if isinstance(obj_dist.velocity_y, RandomValues):
                    logprobs += obj_dist.velocity_y.evaluate_logprobs(
                        obj.velocity_y)
                else:
                    logprobs += 0 if obj_dist.velocity_y == obj.velocity_y else LOG_IMPOSSIBLE_VALUE

        if isinstance(obj_dist.deleted, RandomValues):
            logprobs += 100 * obj_dist.deleted.evaluate_logprobs(
                obj.deleted)
        else:
            logprobs += 0 if obj_dist.deleted == obj.deleted else LOG_IMPOSSIBLE_VALUE

        if size_change_flag:
            if isinstance(obj_dist.w_change, RandomValues):
                logprobs += obj_dist.w_change.evaluate_logprobs(obj.w_change)
            else:
                logprobs += 0 if obj_dist.w_change == obj.w_change else LOG_IMPOSSIBLE_VALUE

            if isinstance(obj_dist.h_change, RandomValues):
                logprobs += obj_dist.h_change.evaluate_logprobs(obj.h_change)
            else:
                logprobs += 0 if obj_dist.h_change == obj.h_change else LOG_IMPOSSIBLE_VALUE

    for idx in leftover_list1:
        obj_dist = obj_list_dist[idx]
        if isinstance(obj_dist.deleted, RandomValues):
            logprobs += 100 * obj_dist.deleted.evaluate_logprobs(1)
        else:
            logprobs += 0 if obj_dist.deleted == 1 else LOG_IMPOSSIBLE_VALUE

    if len(leftover_list2) > 0:
        # log.warning(f'Object creation should be handled separately -- objs dist:\n{obj_list_dist}\n\ninput objs:\n{obj_list}')
        return LOG_IMPOSSIBLE_VALUE

    return logprobs


def fill_unset_values_with_uniform(obj_list_dist: "ObjList",
                                   size_change_flag: bool = False) -> None:
    """
    Fills unset values (velocity_x, velocity_y, deleted) with uniform RandomValues.
    """
    all_possible_velocities = np.arange(-Constants.MAX_ABS_VELOCITY,
                                        Constants.MAX_ABS_VELOCITY + 1)
    all_possible_size_changes = np.arange(-Constants.MAX_ABS_SIZE_CHANGE,
                                          Constants.MAX_ABS_SIZE_CHANGE + 1)

    found = False

    for obj_dist in obj_list_dist:
        if isinstance(obj_dist.velocity_x, RandomValues) or isinstance(
                obj_dist.velocity_y, RandomValues) or isinstance(
                    obj_dist.deleted, RandomValues):
            found = True

        if not isinstance(obj_dist.velocity_x, RandomValues):
            obj_dist.velocity_x = RandomValues(all_possible_velocities)

        if not isinstance(obj_dist.velocity_y, RandomValues):
            obj_dist.velocity_y = RandomValues(all_possible_velocities)

        if not isinstance(obj_dist.deleted, RandomValues):
            obj_dist.deleted = RandomValues(np.arange(2))

        if size_change_flag:
            if isinstance(obj_dist.w_change, RandomValues) or isinstance(
                    obj_dist.h_change, RandomValues):
                found = True

            if not isinstance(obj_dist.w_change, RandomValues):
                obj_dist.w_change = RandomValues(all_possible_size_changes)

            if not isinstance(obj_dist.h_change, RandomValues):
                obj_dist.h_change = RandomValues(all_possible_size_changes)

    return found


def add_noise_to_obj_list_dist(obj_list_dist: "ObjList",
                               make_uniform: bool = False,
                               size_change_flag: bool = False) -> "ObjList":
    """
    Adds noise (or uniform distribution if make_uniform) to every RandomValues in an ObjList.
    """
    log_noise_value = LOG_NOISE_VALUE if not make_uniform else 0
    log_noise_value_deleted = LOG_NOISE_VALUE_DELETED if not make_uniform else 0
    for obj_dist in obj_list_dist:
        if isinstance(obj_dist.velocity_x, RandomValues):
            obj_dist.velocity_x = add_noise_to_random_values(
                obj_dist.velocity_x, -Constants.MAX_ABS_VELOCITY,
                Constants.MAX_ABS_VELOCITY, log_noise_value)

        if isinstance(obj_dist.velocity_y, RandomValues):
            obj_dist.velocity_y = add_noise_to_random_values(
                obj_dist.velocity_y, -Constants.MAX_ABS_VELOCITY,
                Constants.MAX_ABS_VELOCITY, log_noise_value)

        if isinstance(obj_dist.deleted, RandomValues):
            obj_dist.deleted = add_noise_to_random_values(
                obj_dist.deleted, 0, 1, log_noise_value_deleted)

        if size_change_flag:
            if isinstance(obj_dist.w_change, RandomValues):
                obj_dist.w_change = add_noise_to_random_values(
                    obj_dist.w_change, -Constants.MAX_ABS_SIZE_CHANGE,
                    Constants.MAX_ABS_SIZE_CHANGE, log_noise_value)

            if isinstance(obj_dist.h_change, RandomValues):
                obj_dist.h_change = add_noise_to_random_values(
                    obj_dist.h_change, -Constants.MAX_ABS_SIZE_CHANGE,
                    Constants.MAX_ABS_SIZE_CHANGE, log_noise_value)

    return obj_list_dist


def pad_obj_list_dists(obj_list_dists, size_change_flag=False):
    all_possible_velocities = np.arange(-Constants.MAX_ABS_VELOCITY,
                                        Constants.MAX_ABS_VELOCITY + 1)
    all_possible_size_changes = np.arange(-Constants.MAX_ABS_SIZE_CHANGE,
                                          Constants.MAX_ABS_SIZE_CHANGE + 1)

    # all possible locations of the objects
    all_xy_dict = {}
    for obj_list_dist in obj_list_dists:
        for obj in obj_list_dist:
            all_xy_dict[(obj.prev_x, obj.prev_y)] = obj.obj_type

    new_obj_list_dists = []
    for obj_list_dist in obj_list_dists:
        new_obj_list_dist = obj_list_dist

        # location of the objects in the distribution
        xy_dict = {}
        for obj in obj_list_dist:
            xy_dict[(obj.prev_x, obj.prev_y)] = obj.obj_type

        # add half-visible objects
        for xy in all_xy_dict:
            if xy not in xy_dict:
                new_obj_list_dist = new_obj_list_dist.create_object(
                    all_xy_dict[xy], xy[0], xy[1])

        # add noise to the ghost objects
        for obj in new_obj_list_dist:
            if (obj.prev_x, obj.prev_y) not in xy_dict:
                obj.velocity_x = RandomValues(all_possible_velocities)
                obj.velocity_y = RandomValues(all_possible_velocities)
                # When you pad, don't express opinions
                # TODO This can gets tricky -- if we do this, and the programs that have the object get zero weight
                # Then, the objects will take the uniform value as opposed to being
                # Did experiment to skip low weights, but things get even trickier :( -- look into this further
                obj.deleted = RandomValues([0, 1])
                if size_change_flag:
                    obj.w_change = RandomValues(all_possible_size_changes)
                    obj.h_change = RandomValues(all_possible_size_changes)
        new_obj_list_dists.append(new_obj_list_dist)

    return new_obj_list_dists


def combine_obj_list_dists(obj_list_dists: Sequence["ObjList"],
                           weights: Union[np.ndarray, torch.Tensor],
                           use_torch: bool = False,
                           padding: bool = True,
                           size_change_flag: bool = False) -> "ObjList":
    """
    Compute bine the distributions of the objects in the list to a single one with corresponding weight
    padding: if True, it pads the distributions with ghost objects to make them the same length
    """
    if use_torch:
        indices = [
            idx for idx, (obj_list_dist,
                          weight) in enumerate(zip(obj_list_dists, weights))
            if len(obj_list_dist) != 0 and weight != 0
        ]
        if len(indices) == 0:
            return ObjList([])
        obj_list_dists = [
            obj_list_dist for idx, obj_list_dist in enumerate(obj_list_dists)
            if idx in indices
        ]
        weights = weights[indices]
    else:
        tmp = [(obj_list_dist, weight)
               for obj_list_dist, weight in zip(obj_list_dists, weights)
               if len(obj_list_dist) != 0 and weight != 0]
        if len(tmp) == 0:
            return ObjList([])
        obj_list_dists, weights = list(list(zip(*tmp))[0]), np.array(
            list(zip(*tmp))[1])

    if padding:
        # Replace old list of obj dists with the new one
        obj_list_dists = pad_obj_list_dists(obj_list_dists,
                                            size_change_flag=size_change_flag)

    len_obj_list = len(obj_list_dists[0])
    new_objs = []
    for j in range(len_obj_list):
        random_values_x_lst = []
        random_values_y_lst = []
        random_values_deleted_lst = []
        for i in range(len(obj_list_dists)):
            random_values_x_lst.append(obj_list_dists[i][j].velocity_x)
            random_values_y_lst.append(obj_list_dists[i][j].velocity_y)
            random_values_deleted_lst.append(obj_list_dists[i][j].deleted)
        velocity_x = combine_random_values(random_values_x_lst, weights,
                                           use_torch)
        velocity_y = combine_random_values(random_values_y_lst, weights,
                                           use_torch)
        deleted = combine_random_values(random_values_deleted_lst, weights,
                                        use_torch)

        new_obj = obj_list_dists[0][j].copy()
        new_obj.velocity_x = velocity_x
        new_obj.velocity_y = velocity_y
        new_obj.deleted = deleted

        # If size flag is True, combine size changes and set it
        if size_change_flag:
            w_change_lst = []
            h_change_lst = []
            for i in range(len(obj_list_dists)):
                w_change_lst.append(obj_list_dists[i][j].w_change)
                h_change_lst.append(obj_list_dists[i][j].h_change)
            w_change = combine_random_values(w_change_lst, weights, use_torch)
            h_change = combine_random_values(h_change_lst, weights, use_torch)
            new_obj.w_change = w_change
            new_obj.h_change = h_change

        new_objs.append(new_obj)

    return ObjList(new_objs, no_copy=True)


def match_two_obj_lists_helper(obj_list1, obj_list2, list1_hsh, list2_hsh):
    good_pairs = []
    for i, obj1 in enumerate(obj_list1):
        if obj1.id != -1:
            continue
        # Trying to match newly created object
        # Just match by obj_type
        for j, obj2 in enumerate(obj_list2):
            if j not in list2_hsh and obj1.obj_type == obj2.obj_type and obj1.prev_xy == obj2.prev_xy:  # Try to match objects with same location first
                good_pairs.append((i, j))
                list1_hsh[i] = True
                list2_hsh[j] = True
                break
    return good_pairs


def match_two_obj_lists(
    obj_list1: "ObjList", obj_list2: "ObjList"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Matches objects in two ObjLists by ID or obj_type, returning good pairs and leftover indices.
    """
    list1_hsh = {}
    list2_hsh = {}
    good_pairs = []

    # Assumption: output from world model is obj_list1 and newly created objects all have id = -1
    for i, obj1 in enumerate(obj_list1):
        if obj1.id == -1:
            continue
        for j, obj2 in enumerate(obj_list2):
            if obj1.id == obj2.id:
                good_pairs.append((i, j))
                list1_hsh[i] = True
                list2_hsh[j] = True
                break

    if len(list1_hsh) != len(obj_list1):
        good_pairs = good_pairs + match_two_obj_lists_helper(
            obj_list1, obj_list2, list1_hsh, list2_hsh)

    if len(list2_hsh) != len(obj_list2):
        good_pairs = good_pairs + match_two_obj_lists_helper(
            obj_list2, obj_list1, list2_hsh, list1_hsh)

    leftover_list1 = [i for i in range(len(obj_list1)) if i not in list1_hsh]
    leftover_list2 = [i for i in range(len(obj_list2)) if i not in list2_hsh]
    return good_pairs, leftover_list1, leftover_list2


def replace_objs_w_specified_types(obj_list, new_obj_list, obj_types):
    if obj_types is None:
        return new_obj_list
    objs = new_obj_list.objs
    for o in obj_list:
        # Only take objects that are not in obj_types
        if o.obj_type not in obj_types:
            objs.append(o)
    return ObjList(objs, no_copy=True)


def are_two_obj_lists_equal(obj_list1, obj_list2):
    if len(obj_list1) != len(obj_list2):
        return False

    try:
        obj_list1_sorted = sorted(obj_list1.objs, key=lambda obj: (obj.obj_type, obj.x, obj.y))
        obj_list2_sorted = sorted(obj_list2.objs, key=lambda obj: (obj.obj_type, obj.x, obj.y))
        
        for obj1, obj2 in zip(obj_list1_sorted, obj_list2_sorted):
            if obj1.x != obj2.x or obj1.y != obj2.y:
                return False

        return True
    except Exception as e:
        log.warning(f"Error comparing obj lists: {e}")
        return False
