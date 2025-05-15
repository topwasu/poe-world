from abc import ABC, abstractmethod

from classes.helper import *


class GameStateTracker(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init(self, atari_env):
        pass

    @abstractmethod
    def update(self, atari_env, game_state):
        pass


class GenericGameStateTracker(GameStateTracker):
    def __init__(self):
        pass

    def init(self, atari_env):
        return GameState.RESTART

    def update(self, atari_env, game_state):
        return GameState.NORMAL


class MontezumaRevengeStateTracker():
    def __init__(self):
        self.lives = None

    def init(self, atari_env):
        obj_list = ObjList(
            [Obj(o, id=idx) for idx, o in enumerate(atari_env.env.objects)])
        self.lives = len(obj_list.get_objs_by_obj_type('life')) + len(
            obj_list.get_objs_by_obj_type('lifecount'))
        return GameState.RESTART

    def update(self, atari_env, game_state):
        game_state, cur_lives = self._update_helper(atari_env.env, atari_env.object_tracker, game_state, self.lives)
        self.lives = cur_lives
        return game_state
    
    def _update_helper(self, env, object_tracker, game_state, cur_lives):
        obj_list = ObjList([Obj(o) for o in env.objects])

        # Assumption: we are skipping the dying phase
        if game_state == GameState.NORMAL:
            lives = len(obj_list.get_objs_by_obj_type('life')) + len(
                obj_list.get_objs_by_obj_type('lifecount'))
            if lives < cur_lives or (cur_lives == 0 and env.get_ram()[112] == 15):
                game_state = GameState.DEAD
            cur_lives = lives
        elif game_state == GameState.DEAD:
            if env.get_ram()[2] == 5:
                env.reset()
                return GameState.GAMEOVER, cur_lives
            else:
                while env.get_ram()[2] != 4:
                    if env.get_ram()[2] == 5:
                        env.reset()
                        return GameState.GAMEOVER, cur_lives
                    env.step(0)
                    # atari_env.env.render()
                if env.get_ram()[2] != 5:
                    env.step(0)

            game_state = GameState.RESTART
        elif game_state == GameState.RESTART:
            game_state = GameState.NORMAL
        return game_state, cur_lives


class PitfallStateTracker(GameStateTracker):
    def __init__(self):
        self.lives = None

    def init(self, atari_env):
        obj_list = ObjList(
            [Obj(o, id=idx) for idx, o in enumerate(atari_env.env.objects)])
        self.lives = len(obj_list.get_objs_by_obj_type('life')) + len(
            obj_list.get_objs_by_obj_type('lifecount'))
        return GameState.RESTART

    def update(self, atari_env, game_state):
        # TODO: Double check this implementation
        if game_state == GameState.NORMAL:
            if atari_env.env.get_ram()[30] > 0:
                game_state = GameState.DEAD
        elif game_state == GameState.DEAD:
            while atari_env.env.get_ram()[30] != 0 or (
                    atari_env.env.get_ram()[105] != 0
                    and atari_env.env.get_ram()[105] != 32):
                obj_list = ObjList([Obj(o) for o in atari_env.env.objects])
                atari_env.object_tracker.update(obj_list)

                atari_env.env.step(0)
            game_state = GameState.RESTART
        elif game_state == GameState.RESTART:
            game_state = GameState.NORMAL

        obj_list = ObjList(
            [Obj(o, id=idx) for idx, o in enumerate(atari_env.env.objects)])
        return game_state


class PongStateTracker(GameStateTracker):
    def __init__(self):
        self.lives = None

    def init(self, atari_env):
        return GameState.NORMAL

    def update(self, atari_env, game_state):
        return self._update_helper(atari_env.env, atari_env.object_tracker, game_state)
    
    def _update_helper(self, env, object_tracker, game_state):
        if env.get_ram()[13] > 20 or env.get_ram()[14] > 20:
            return GameState.GAMEOVER
        return GameState.NORMAL
    
    
class MontezumaRevengeAltStateTracker(MontezumaRevengeStateTracker):
    def __init__(self):
        super().__init__()
        
    def init(self, atari_env):
        self.lives = [5, 5, 4]
        return GameState.RESTART
        
    def update(self, atari_env, game_state):
        game_state1, cur_lives1 = self._update_helper(atari_env.env, atari_env.object_tracker, GameState.DEAD if self.lives[0] != 5 else GameState.NORMAL, self.lives[0])
        game_state2, cur_lives2 = self._update_helper(atari_env.env2, atari_env.object_tracker, GameState.DEAD if self.lives[1] != 5 else GameState.NORMAL, self.lives[1])
        game_state3, cur_lives3 = self._update_helper(atari_env.env3, atari_env.object_tracker, GameState.DEAD if self.lives[2] != 4 else GameState.NORMAL, self.lives[2])
        
        if game_state1 == GameState.RESTART or game_state2 == GameState.RESTART or game_state3 == GameState.RESTART:
            self.lives = [5, 5, 4]
            return GameState.GAMEOVER
        
        self.lives = [cur_lives1, cur_lives2, cur_lives3]
        return GameState.NORMAL
        
    
class PongAltStateTracker(PongStateTracker):
    def __init__(self):
        super().__init__()
        
    def update(self, atari_env, game_state):
        game_state1 = self._update_helper(atari_env.env, atari_env.object_tracker, game_state)
        game_state2 = self._update_helper(atari_env.env2, atari_env.object_tracker, game_state)
        
        if game_state1 == GameState.GAMEOVER or game_state2 == GameState.GAMEOVER:
            return GameState.GAMEOVER
        return GameState.NORMAL
