import time
from typing import Tuple, Optional, List, Any, Union

from ocatari.core import OCAtari

from classes.helper import *
from classes.game_utils import *
from classes.envs.object_tracker import *
from classes.envs.game_state_tracker import *
from data.atari import load_atari_observations


def get_goal_obj_type_by_game(config):
    if config.task.startswith('MontezumaRevenge'):
        return 'key'
    elif config.task.startswith('Pong'):
        return 'ball'
    else:
        raise NotImplementedError


def create_atari_env(config, env_name: str, 
                     renderer: Optional[Any] = None, 
                     image_renderer: Optional[Any] = None, 
                     skip_gameover_if_possible: bool = True, 
                     **kwargs) -> 'AtariEnv':
    """
    Creates and configures an Atari environment based on the game name.
    
    Args:
        env_name: Name of the Atari game to create
        **kwargs: Additional arguments to pass to AtariEnv constructor
    
    Returns:
        Configured AtariEnv instance
    
    Raises:
        NotImplementedError: If env_name is not supported
    """
    if env_name == 'MontezumaRevenge':
        object_tracker = ObjectTracker()
        game_state_tracker = MontezumaRevengeStateTracker()
        actions_enum = MontezumaRevengeActions
        skip_gameover = skip_gameover_if_possible
        frameskip = 3
    elif env_name == 'MontezumaRevengeAlt':
        object_tracker = ObjectTracker()
        game_state_tracker = MontezumaRevengeAltStateTracker()
        actions_enum = MontezumaRevengeActions
        skip_gameover = skip_gameover_if_possible
        frameskip = 3
    elif env_name == 'Pitfall':
        object_tracker = ObjectTracker()
        game_state_tracker = PitfallStateTracker()
        actions_enum = PitfallActions
        skip_gameover = skip_gameover_if_possible
        frameskip = 3
    elif env_name == 'Breakout':
        object_tracker = ObjectTracker()
        game_state_tracker = GenericGameStateTracker()
        actions_enum = BreakoutActions
        skip_gameover = skip_gameover_if_possible
        frameskip = 1
    elif env_name == 'Pong':
        object_tracker = ObjectTracker()
        game_state_tracker = PongStateTracker()
        actions_enum = PongActions
        skip_gameover = False
        frameskip = 3
    elif env_name == 'PongAlt':
        object_tracker = ObjectTracker()
        game_state_tracker = PongAltStateTracker()
        actions_enum = PongActions
        skip_gameover = False
        frameskip = 3
    else:
        raise NotImplementedError
    
    if env_name == 'PongAlt':
        return PongAltEnv(config,
                          env_name,
                          object_tracker,
                          game_state_tracker,
                          actions_enum,
                          frameskip=frameskip,
                          renderer=renderer,
                          image_renderer=image_renderer,
                          skip_gameover=skip_gameover,
                          **kwargs)
    elif env_name == 'MontezumaRevengeAlt':
        return MontezumaAltEnv(config,
                               env_name,
                               object_tracker,
                               game_state_tracker,
                               actions_enum,
                               frameskip=frameskip,
                               renderer=renderer,
                               image_renderer=image_renderer,
                               skip_gameover=skip_gameover,
                               **kwargs)

    return AtariEnv(config,
                    env_name,
                    object_tracker,
                    game_state_tracker,
                    actions_enum,
                    frameskip=frameskip,
                    skip_gameover=skip_gameover,
                    **kwargs)


class AtariEnv:
    """
    Wrapper class for Atari environments that handles object tracking and game state.
    """
    def __init__(self,
                 config,
                 env_name: str,
                 object_tracker: ObjectTracker,
                 game_state_tracker: GameStateTracker,
                 actions_enum: BaseActions,
                 frameskip: int = 3,
                 skip_gameover: bool = False,
                 recorder: Optional[Any] = None) -> None:
        """
        Initialize the Atari environment.

        Args:
            env_name: Name of the Atari game
            object_tracker: Tracker for game objects
            game_state_tracker: Tracker for game state
            actions_enum: Enum class containing possible actions
            frameskip: Number of frames to skip between actions
            recorder: Optional recorder for capturing gameplay
        """
        if env_name not in [
                'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Breakout', 'Pong'
        ]:
            raise NotImplementedError

        self.config = config
        self.env_name = env_name
        self.object_tracker = object_tracker
        self.game_state_tracker = game_state_tracker
        self.actions_enum = actions_enum
        self.frameskip = frameskip
        self.recorder = recorder

        if config.world_model_learner.obj_type == 'beam' or config.world_model_learner.obj_type == 'rope' or config.rope_mode or config.world_model_learner.obj_type == 'disappearingtarpit':
            render_mode = "rgb_array"
        else:
            render_mode = "rgb_array" if self.recorder is not None else "human"
        self.env = OCAtari(env_name,
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)

        # class variables
        self.game_state = None
        self.memory = StateMemory(Constants.MEMORY_LENGTH)
        
        self.n_reset = 0
        self.skip_gameover = skip_gameover

    def reset(self) -> Tuple[ObjListWithMemory, GameState]:
        """
        Resets the environment to initial state.

        Returns:
            Tuple of (object list, game state)
        """
        self.n_reset += 1
        self.env.reset()

        if self.recorder is not None:
            self.recorder.record(self)
        else:
            self.env.render()

        self.obj_list = ObjList(
            [Obj(o, id=idx) for idx, o in enumerate(self.env.objects)])

        self.object_tracker.update(self.obj_list)

        self.game_state = self.game_state_tracker.init(self)
        
        self.memory.reset()

        return ObjListWithMemory(self.obj_list, self.memory), self.game_state

    def step(self, action: str) -> Tuple[ObjListWithMemory, GameState]:
        """
        Perform a step in the environment based on the given action.

        Args:
            action: Action string to execute

        Returns:
            Tuple of (object list, game state)
        """
        # upper case the action and remove anything that is not alphabetic
        action = "".join([c for c in action.upper() if c.isalpha()])
        action_index = self.actions_enum.get_all_possible_actions().index(
            action)
        
        self.memory.add_obj_list_and_action(self.obj_list, action)

        _, reward, _, terminated, _ = self.env.step(action_index)
        self.reward = reward

        if self.recorder is not None:
            self.recorder.record(self)
        else:
            self.env.render()

        self.game_state = self.game_state_tracker.update(self, self.game_state)
        
        if self.game_state == GameState.GAMEOVER and self.skip_gameover:
            self.n_reset += 1
            self.game_state = GameState.RESTART

        self.obj_list = ObjList([Obj(o) for o in self.env.objects])
        self.object_tracker.update(self.obj_list)
        self.object_tracker.handle_game_state(self.obj_list, self.game_state)
            
        return ObjListWithMemory(self.obj_list, self.memory), self.game_state

    def get_actions_enum(self):
        return self.actions_enum

    def get_keys2actions(self):
        return self.env.unwrapped.get_keys_to_action()


class ImaginedAtariEnv():
    def __init__(self, config, world_model, object_tracker: ObjectTracker,
                 renderer):
        self.config = config
        env_name = config.task
        self.env_name = env_name
        self.world_model = world_model
        self.object_tracker = object_tracker
        self.renderer = renderer
        
        atari_env = create_atari_env(config, env_name)

        self.init_obj_list, self.init_game_state = atari_env.reset()
        self.object_tracker.update(self.init_obj_list)
        
        if config.imagined_atari_env.start_at != 0:
            _, actions, _ = load_atari_observations(config.task.replace('Alt', '') + config.obs_suffix)
            for i in range(config.imagined_atari_env.start_at):
                self.init_obj_list, self.init_game_state = atari_env.step(actions[i])
                self.object_tracker.update(self.init_obj_list)
            
        self.init_object_tracker = copy.deepcopy(self.object_tracker)

        self.actions_enum = atari_env.get_actions_enum()
        self.keys2actions = atari_env.env.unwrapped.get_keys_to_action()

        self.obj_list = self.init_obj_list.deepcopy()

        self.memory = StateMemory(Constants.MEMORY_LENGTH)

    def reset(self) -> Tuple[ObjListWithMemory, Optional[GameState]]:
        """
        Resets the imagined environment.

        Returns:
            Tuple of (object list, game state)
        """
        self.obj_list = self.init_obj_list.deepcopy()

        self.object_tracker = copy.deepcopy(self.init_object_tracker)
        # if self.config.world_model_learner.obj_type == 'beam' \
        #     or self.config.world_model_learner.obj_type == 'rope' \
        #         or self.config.rope_mode or self.config.imagined_atari_env.start_at != 0:
        #     self.object_tracker = copy.deepcopy(self.init_object_tracker)
        # else:
        #     self.object_tracker.reset()

        self.renderer.render(self.obj_list)

        self.memory.reset()

        return ObjListWithMemory(self.obj_list, self.memory), self.init_game_state

    def step(self, action: str) -> Tuple[ObjListWithMemory, None]:
        """
        Simulates a step using the world model.

        Args:
            action: Action string to simulate

        Returns:
            Tuple of (predicted object list, None)
        """

        new_obj_list = self.world_model.sample_next_scene(
            self.obj_list,
            action,
            memory=self.memory,
            det=self.config.det_world_model)
        
        self.memory.add_obj_list_and_action(self.obj_list, action)
        self.obj_list = new_obj_list

        self.object_tracker.update(self.obj_list)

        self.renderer.render(self.obj_list)
        
        time.sleep(self.config.imagined_atari_env.time_lag)
        
        return ObjListWithMemory(self.obj_list, self.memory), None

    def get_actions_enum(self):
        return self.actions_enum

    def get_keys2actions(self):
        return self.keys2actions



class PongAltEnv(AtariEnv):
    """
    Wrapper class for Atari environments that handles object tracking and game state.
    """
    def __init__(self,
                 config,
                 env_name: str,
                 object_tracker: ObjectTracker,
                 game_state_tracker: GameStateTracker,
                 actions_enum: BaseActions,
                 frameskip: int = 3,
                 skip_gameover: bool = False,
                 recorder: Optional[Any] = None,
                 renderer: Optional[Any] = None,
                 image_renderer: Optional[Any] = None) -> None:
        """
        Initialize the Atari environment.

        Args:
            env_name: Name of the Atari game
            object_tracker: Tracker for game objects
            game_state_tracker: Tracker for game state
            actions_enum: Enum class containing possible actions
            frameskip: Number of frames to skip between actions
            recorder: Optional recorder for capturing gameplay
        """
        if env_name not in [
                'PongAlt'
        ]:
            raise NotImplementedError

        self.config = config
        self.env_name = env_name
        self.object_tracker = object_tracker
        self.game_state_tracker = game_state_tracker
        self.actions_enum = actions_enum
        self.frameskip = frameskip
        self.recorder = recorder
        self.skip_gameover = skip_gameover
        
        if self.skip_gameover:
            raise NotImplementedError

        if config.world_model_learner.obj_type == 'beam' or config.world_model_learner.obj_type == 'rope' or config.rope_mode or config.world_model_learner.obj_type == 'disappearingtarpit':
            render_mode = "rgb_array"
        else:
            render_mode = "rgb_array" if self.recorder is not None else "human"
        self.env = OCAtari('Pong',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)
        self.env2 = OCAtari('Pong',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)
        self.env3 = OCAtari('Pong',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)
        
        self.difficulty = 'hard'

        # class variables
        self.game_state = None
        self.memory = StateMemory(Constants.MEMORY_LENGTH)
        
        # set renderer
        self.renderer = renderer
        self.image_renderer = image_renderer
        self.env.reset()
        self.env2.reset()
        self.env3.reset()
        for i in range(30): 
            self.env.step(0)
        for i in range(50): 
            self.env2.step(0)
        for i in range(70): 
            self.env3.step(0)
        self.cum_wins = 0
        self.cum_losses = 0
            
        self.init_obj_list = self._grab_obj_list()
        
        self.reward = 0

    def reset(self) -> Tuple[ObjListWithMemory, GameState]:
        """
        Resets the environment to initial state.

        Returns:
            Tuple of (object list, game state)
        """
        # Can optimize this by not resetting the envs
        # But this is easier
        self.env.reset()
        self.env2.reset()
        self.env3.reset()
        for i in range(30): 
            self.env.step(0)
        for i in range(50): 
            self.env2.step(0)
        for i in range(70): 
            self.env3.step(0)
        self.cum_wins = 0
        self.cum_losses = 0
        self.reward = 0
            
        self.memory.reset()

        self.obj_list = self._grab_obj_list()
        # Remove tags from the object list after already tracked the object list
        self.obj_list_no_tags = self._remove_tags(self.obj_list)
        
        self.object_tracker.reset()
        self.object_tracker.update(self.obj_list_no_tags, handle_same_id=True)
        self.object_tracker.handle_game_state(self.obj_list_no_tags, self.game_state)
        self.game_state = self.game_state_tracker.init(self)
        
        # Render the no-tags object list
        if self.recorder is not None:
            self.recorder.frames.append(self.image_renderer.render(self.obj_list_no_tags))
        else:
            if self.renderer is not None:
                self.renderer.render(self.obj_list_no_tags)

        return ObjListWithMemory(self.obj_list_no_tags, self.memory), self.game_state

    def step(self, action: str) -> Tuple[ObjListWithMemory, GameState]:
        """
        Perform a step in the environment based on the given action.

        Args:
            action: Action string to execute

        Returns:
            Tuple of (object list, game state)
        """
        if self.game_state == GameState.GAMEOVER:
            self.reward = 0
            return ObjListWithMemory(self.obj_list_no_tags, self.memory), self.game_state
        
        # upper case the action and remove anything that is not alphabetic
        action = "".join([c for c in action.upper() if c.isalpha()])
        action_index = self.actions_enum.get_all_possible_actions().index(
            action)
        
        self.memory.add_obj_list_and_action(self.obj_list, action)

        _, reward1, _, _, _ = self.env.step(action_index)
        _, reward2, _, _, _ = self.env2.step(action_index)
        if self.difficulty == 'hard':
            _, reward3, _, _, _ = self.env3.step(action_index)
        else:
            reward3 = 0
        
        self.cum_wins += max(0, reward1) + max(0, reward2) + max(0, reward3)
        self.cum_losses += max(0, -reward1) + max(0, -reward2) + max(0, -reward3)
        self.reward = reward1 + reward2 + reward3
        
        self.obj_list = self._grab_obj_list()
        # Remove tags from the object list after already tracked the object list
        self.obj_list_no_tags = self._remove_tags(self.obj_list)
        
        self.object_tracker.update(self.obj_list_no_tags, handle_same_id=True)
        self.object_tracker.handle_game_state(self.obj_list_no_tags, self.game_state)
        self.game_state = self.game_state_tracker.update(self, self.game_state)
        
        # Render the no-tags object list
        if self.recorder is not None:
            self.recorder.frames.append(self.image_renderer.render(self.obj_list_no_tags))
        else:
            if self.renderer is not None:
                self.renderer.render(self.obj_list_no_tags)
                
        if self.cum_wins > 20 or self.cum_losses > 20:
            self.obj_list_no_tags = ObjList([])
            self.game_state = GameState.GAMEOVER

        return ObjListWithMemory(self.obj_list_no_tags, self.memory), self.game_state

    def get_actions_enum(self):
        return self.actions_enum

    def get_keys2actions(self):
        return self.env.unwrapped.get_keys_to_action()
    
    def _grab_obj_list(self):
        all_objs = []
        for idx, o in enumerate(self.env.objects):
            obj = Obj(o, id=len(all_objs))
            all_objs.append(obj)
        for idx, o in enumerate(self.env2.objects):
            obj = Obj(o, id=len(all_objs))
            if obj.obj_type not in ['enemy', 'ball']:
                continue
            obj.obj_type = obj.obj_type + '_2'
            if obj.obj_type == 'enemy_2':
                obj.prev_x += 1e-6
            # obj.prev_x = 160 * 2 - obj.prev_x - obj.w
            # obj.velocity_x = -obj.velocity_x
            all_objs.append(obj)
            
        if self.difficulty == 'hard':
            for idx, o in enumerate(self.env3.objects):
                obj = Obj(o, id=len(all_objs))
                if obj.obj_type not in ['enemy', 'ball']:
                    continue
                obj.obj_type = obj.obj_type + '_3'
                if obj.obj_type == 'enemy_3':
                    obj.prev_x += 2e-6
                all_objs.append(obj)
                
        return ObjList(all_objs, no_copy=True)
    
    def _remove_tags(self, obj_list):
        all_objs = []
        for obj in obj_list:
            obj = obj.copy()
            obj.obj_type = obj.obj_type.split('_')[0]
            all_objs.append(obj)
        return ObjList(all_objs, no_copy=True)
    
    
class MontezumaAltEnv(AtariEnv):
    """
    Wrapper class for Atari environments that handles object tracking and game state.
    """
    def __init__(self,
                 config,
                 env_name: str,
                 object_tracker: ObjectTracker,
                 game_state_tracker: GameStateTracker,
                 actions_enum: BaseActions,
                 frameskip: int = 3,
                 skip_gameover: bool = False,
                 recorder: Optional[Any] = None,
                 renderer: Optional[Any] = None,
                 image_renderer: Optional[Any] = None) -> None:
        """
        Initialize the Atari environment.

        Args:
            env_name: Name of the Atari game
            object_tracker: Tracker for game objects
            game_state_tracker: Tracker for game state
            actions_enum: Enum class containing possible actions
            frameskip: Number of frames to skip between actions
            recorder: Optional recorder for capturing gameplay
        """
        if env_name not in [
                'MontezumaRevengeAlt'
        ]:
            raise NotImplementedError

        self.config = config
        self.env_name = env_name
        self.object_tracker = object_tracker
        self.game_state_tracker = game_state_tracker
        self.actions_enum = actions_enum
        self.frameskip = frameskip
        self.recorder = recorder
        self.skip_gameover = skip_gameover
        if config.world_model_learner.obj_type == 'beam' or config.world_model_learner.obj_type == 'rope' or config.rope_mode or config.world_model_learner.obj_type == 'disappearingtarpit':
            render_mode = "rgb_array"
        else:
            render_mode = "rgb_array" if self.recorder is not None else "human"
        self.env = OCAtari('MontezumaRevenge',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)
        self.env2 = OCAtari('MontezumaRevenge',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)
        self.env3 = OCAtari('MontezumaRevenge',
                           mode="revised",
                           hud=True,
                           render_mode=render_mode,
                           render_oc_overlay=True,
                           frameskip=frameskip)

        # class variables
        self.game_state = None
        self.memory = StateMemory(Constants.MEMORY_LENGTH)
        
        # set renderer
        self.renderer = renderer
        self.image_renderer = image_renderer
        
        # self.actions = ['DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWNLEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFTFIRE', 'LEFTFIRE', 'LEFTFIRE', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'UP', 'DOWN', 'DOWN', 'DOWN']
        self.actions = ['DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT']
        self.env.reset()
        self.env2.reset()
        self.env3.reset()
        for a in self.actions:
            self.env.step(self.actions_enum.get_all_possible_actions().index(a))
        for a in self.actions + ['NOOP'] * 100:
            self.env2.step(self.actions_enum.get_all_possible_actions().index(a))
            
        for a in self.actions + ['LEFT'] * 20 + ['NOOP'] * 20 + self.actions:
        # for a in self.actions + ['NOOP'] * 20:
            self.env3.step(self.actions_enum.get_all_possible_actions().index(a))
            
        self.init_obj_list = self._grab_obj_list()
        
        self.reward = 0
        self.n_reset = 0

    def reset(self) -> Tuple[ObjListWithMemory, GameState]:
        """
        Resets the environment to initial state.

        Returns:
            Tuple of (object list, game state)
        """
        self.n_reset += 1
        # Can optimize this by not resetting the envs
        # But this is easier
        self.env.reset()
        self.env2.reset()
        self.env3.reset()
        for a in self.actions:
            self.env.step(self.actions_enum.get_all_possible_actions().index(a))
        for a in self.actions + ['NOOP'] * 100:
            self.env2.step(self.actions_enum.get_all_possible_actions().index(a))
        for a in self.actions + ['LEFT'] * 20 + ['NOOP'] * 20 + self.actions:
        # for a in self.actions + ['NOOP'] * 20:
            self.env3.step(self.actions_enum.get_all_possible_actions().index(a))
            
        self.reward = 0
            
        self.memory.reset()

        self.obj_list = self._grab_obj_list()
        # Remove tags from the object list after already tracked the object list
        self.obj_list_no_tags = self._remove_tags(self.obj_list)
        
        self.object_tracker.reset()
        self.object_tracker.update(self.obj_list_no_tags)
        self.object_tracker.handle_game_state(self.obj_list_no_tags, self.game_state)
        self.game_state = self.game_state_tracker.init(self)
        
        # Render the no-tags object list
        if self.recorder is not None:
            self.recorder.frames.append(self.image_renderer.render(self.obj_list_no_tags))
        else:
            if self.renderer is not None:
                self.renderer.render(self.obj_list_no_tags)

        return ObjListWithMemory(self.obj_list_no_tags, self.memory), self.game_state

    def step(self, action: str) -> Tuple[ObjListWithMemory, GameState]:
        """
        Perform a step in the environment based on the given action.

        Args:
            action: Action string to execute

        Returns:
            Tuple of (object list, game state)
        """
        # upper case the action and remove anything that is not alphabetic
        action = "".join([c for c in action.upper() if c.isalpha()])
        action_indices = self._grab_action_indices(action)
        
        self.memory.add_obj_list_and_action(self.obj_list, action)
        
        _, reward1, _, _, _ = self.env.step(action_indices[0])
        _, reward2, _, _, _ = self.env2.step(action_indices[1])
        _, reward3, _, _, _ = self.env3.step(action_indices[2])
        
        self.reward = reward1 + reward2 + reward3
        
        self.game_state = self.game_state_tracker.update(self, self.game_state)
        
        if self.game_state == GameState.GAMEOVER and self.skip_gameover:
            self.n_reset += 1
            self.game_state = GameState.RESTART
        
        if self.game_state == GameState.RESTART:
            self.env.reset()
            self.env2.reset()
            self.env3.reset()
            for a in self.actions:
                self.env.step(self.actions_enum.get_all_possible_actions().index(a))
            for a in self.actions + ['NOOP'] * 100:
                self.env2.step(self.actions_enum.get_all_possible_actions().index(a))
            for a in self.actions + ['LEFT'] * 20 + ['NOOP'] * 20 + self.actions:
            # for a in self.actions + ['NOOP'] * 20:
                self.env3.step(self.actions_enum.get_all_possible_actions().index(a))
        
        self.obj_list = self._grab_obj_list()
        # Remove tags from the object list after already tracked the object list
        self.obj_list_no_tags = self._remove_tags(self.obj_list)
        
        self.object_tracker.update(self.obj_list_no_tags)
        self.object_tracker.handle_game_state(self.obj_list_no_tags, self.game_state)
            
        if self.recorder is not None:
            self.recorder.frames.append(self.image_renderer.render(self.obj_list_no_tags))
        else:
            if self.renderer is not None:
                self.renderer.render(self.obj_list_no_tags)

        return ObjListWithMemory(self.obj_list_no_tags, self.memory), self.game_state

    def get_actions_enum(self):
        return self.actions_enum

    def get_keys2actions(self):
        return self.env.unwrapped.get_keys_to_action()
    
    def _grab_obj_list(self):
        player_pos = self._determine_player_position()
        all_objs = []
        for idx, o in enumerate(self.env.objects):
            if o.prev_y > 136 or (o.prev_y == 136 and o.category.lower() in ['ladder', 'wall']) or o.category.lower() == 'player':
                if o.category.lower() == 'player' and player_pos != 0:
                    continue
                if o.category.lower() == 'ladder' and o.prev_x != 16:
                    continue
                obj = Obj(o, id=len(all_objs))
                
                # flip
                obj.prev_x = 160 - obj.prev_x - obj.w
                obj.velocity_x = -obj.velocity_x
                
                all_objs.append(obj)
        for idx, o in enumerate(self.env2.objects):
            if o.prev_y > 136 or (o.prev_y == 136 and o.category.lower() in ['ladder', 'wall']) or o.category.lower() == 'player':
                if o.category.lower() == 'player' and player_pos != 1:
                    continue
                if o.category.lower() == 'ladder' and o.prev_x != 16:
                    continue
                obj = Obj(o, id=len(all_objs))
                obj.obj_type = obj.obj_type + '_2'
                
                # move up
                obj.prev_y = obj.prev_y - 44
                
                obj.prev_x = obj.prev_x + 3
                
                if obj.obj_type == 'wall_2' and obj.prev_x == 0 + 3:
                    obj.prev_x = 0
                    obj.w = obj.w + 3
                elif obj.obj_type == 'wall_2' and obj.prev_x == 140 + 3:
                    obj.w = obj.w - 3
                
                all_objs.append(obj)
        for idx, o in enumerate(self.env3.objects):
            if o.prev_y > 136 or (o.prev_y == 136 and o.category.lower() in ['ladder', 'wall']) or (o.prev_y > 93 and o.prev_x < 16) or o.category.lower() == 'player':
                if o.category.lower() == 'player' and player_pos != 2:
                    continue
                if o.category.lower() == 'ladder' and o.prev_x != 16:
                    continue
                obj = Obj(o, id=len(all_objs))
                obj.obj_type = obj.obj_type + '_3'
                
                # flip
                obj.prev_x = 160 - obj.prev_x - obj.w
                obj.velocity_x = -obj.velocity_x
                
                # move up
                obj.prev_y = obj.prev_y - 44 * 2
                
                all_objs.append(obj)
        return ObjList(all_objs, no_copy=True)
    
    def _determine_player_position(self, ret_pos_x=False):
        player_o = self.env.objects[0]
        player_o_2 = self.env2.objects[0]
        player_o_3 = self.env3.objects[0]
        pos = None
        if player_o.y + player_o.h > 136:
            pos = 0
            x = player_o.x
        elif player_o_2.y + player_o_2.h > 136:
            pos = 1
            x = player_o_2.x
        else:
            pos = 2
            x = player_o_3.x
        
        
        if ret_pos_x:
            return pos, x
        else:
            return pos
        
    def _grab_action_indices(self, action):
        player_pos, x = self._determine_player_position(ret_pos_x=True)
        
        if x > 100:
            action = action.replace('UP', '')
            if action == '':
                action = 'NOOP'
        
        action_index = self.actions_enum.get_all_possible_actions().index(
            action)
        flipped_action_index = self.actions_enum.get_all_possible_actions().index(
            action.replace('RIGHT', 'DUMMY').replace('LEFT', 'RIGHT').replace('DUMMY', 'LEFT'))
        
        if player_pos > 0 and action == 'DOWN' and x >= 125:
            # going right to touch the wall again
            return flipped_action_index if player_pos == 1 else 0, action_index if player_pos == 2 else (0 if self.env2.objects[0].x == 129 else (4 if self.env2.objects[0].x == 132 else 3)), (0 if self.env3.objects[0].x == 129 else (4 if self.env3.objects[0].x == 132 else 3))
            

        return flipped_action_index if player_pos == 0 else 0, action_index if player_pos == 1 else (0 if self.env2.objects[0].x == 129 or self.env2.objects[0].x == 20 else (4 if self.env2.objects[0].x == 132 else 3)), flipped_action_index if player_pos == 2 else (0 if self.env3.objects[0].x == 129 or self.env3.objects[0].x == 20 else (4 if self.env3.objects[0].x == 132 else 3))
    
    def _remove_tags(self, obj_list):
        all_objs = []
        for obj in obj_list:
            obj = obj.copy()
            obj.obj_type = obj.obj_type.split('_')[0]
            all_objs.append(obj)
        return ObjList(all_objs, no_copy=True)