import numpy as np
import time
import traceback

from classes.helper import *
from classes.envs.env import AtariEnv


class EnvPlayer():
    """
    EnvPlayer class enables user interaction with game environments
    Handles keyboard input mapping to game actions and manages game state
    """
    def __init__(self, env: AtariEnv):
        """
        Initialize environment player with game-specific controls
        Args:
            env: Game environment instance
        """
        self.env = env

        self.paused = False
        self.current_keys_down = set()  # Track currently pressed keys
        # # Define key-to-action mappings for different games
        # if env.env_name == 'Pong':
        #     # Map keys for Pong: None=0, d=2--right, a=3--left
        #     self.keys2actions = {(None, ): 0, (100, ): 2, (97, ): 3}
        # elif env.env_name == "Breakout":
        #     # Map keys for Breakout: None=0, Space=1, D=2, A=3
        #     self.keys2actions = {(None, ): 0, (32, ): 1, (100, ): 2, (97, ): 3}
        # else:
        #     # Use environment's default key mapping
        self.keys2actions = self.env.get_keys2actions()

        self.running = True
        self.actions_enum = self.env.get_actions_enum()

    def run(self, existing_actions=[], slow=False):
        """
        Main game loop that handles environment steps and user input
        Args:
            existing_actions: List of predefined actions to replay
            slow: Boolean to add delay between actions
        Returns:
            observations: List of game observations
            actions: List of actions taken
            game_states: List of game states
        """
        obj_list, game_state = self.env.reset()
        observations, actions, game_states = [obj_list], [], [game_state]
        idx = 0
        while self.running:
            try:
                input = self._handle_user_input()

                if not self.paused:
                    if input == 'RESET':
                        obj_list, game_state = self.env.reset()
                        actions.append('RESTART')
                    else:
                        # Either use existing actions or get new action from user input
                        if idx < len(existing_actions):
                            action = existing_actions[idx]
                            idx += 1
                        else:
                            if slow:
                                time.sleep(
                                    0.1)  # Add delay if slow mode is enabled
                            action_index = self._get_action()
                            action = self.actions_enum.get_all_possible_actions(
                            )[action_index]
                        actions.append(action)

                        if action == 'RESTART':
                            obj_list, game_state = self.env.reset()
                        else:
                            obj_list, game_state = self.env.step(action)

                observations.append(obj_list)
                game_states.append(game_state)
            except:
                log.info(traceback.format_exc())
                break
        pygame.quit()
        # Ensure actions list matches observations length
        if len(observations) == len(actions):
            actions = actions[:-1]
        return observations, actions, game_states

    def _get_action(self):
        """
        Convert currently pressed keys to corresponding game action
        Returns:
            int: Action index based on pressed keys
        """
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            action = self.keys2actions[pressed_keys]
            # logging.debug(f'Pressed key: {pressed_keys}, Action: {action}')
            return action
        else:
            return 0  # Default action if key combination is not mapped

    def _handle_user_input(self):
        """
        Process pygame events and update game state accordingly
        Handles window closing, key presses/releases, and game controls
        Returns:
            str: 'RESET' if reset command issued, None otherwise
        """
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # P key toggles pause
                    self.paused = not self.paused

                if event.key == pygame.K_r:  # R key resets game
                    self.env.reset()
                    return 'RESET'

                if event.key == pygame.K_m:  # M key for object inspection
                    objects = self.env.objects

                if (event.key, ) in self.keys2actions.keys(
                ):  # Add valid game control keys
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:
                if (event.key,
                    ) in self.keys2actions.keys():  # Remove released keys
                    self.current_keys_down.remove(event.key)
