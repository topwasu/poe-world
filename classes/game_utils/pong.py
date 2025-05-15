from classes.game_utils.base_game import BaseActions

pong_wh_dict = {
    'player': (4, 15),  # Paddle dimensions (width, height)
    'ball': (2, 4),  # Ball dimensions
    'enemy': (4, 15),  # Enemy paddle dimensions
    'player_score': (12, 20),  # Score display area
    'enemy_score': (12, 20),  # Score display area
    'wall': (148, 5),
    'zone': (5, 169),
}

PONG_MAX_ABS_VELOCITY = 30  # Maximum ball velocity

PONG_HISTORY_LENGTH = 100

PONG_MAX_ABS_SIZE_CHANGE = 1

PONG_ACTIONS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


class PongActions(BaseActions):
    NOOP = 'NOOP'  # No operation
    FIRE = 'FIRE'  # Start the game
    RIGHT = 'RIGHT'  # Move the paddle to the right
    LEFT = 'LEFT'  # Move the paddle to the left
    RIGHTFIRE = 'RIGHTFIRE'  # Move the paddle to the right and start the game
    LEFTFIRE = 'LEFTFIRE'  # Move the paddle to the left and start the game

    @staticmethod
    def get_all_possible_actions():
        return ['NOOP', 'FIRE', 'RIGHT', 'LEFT', "RIGHTFIRE", "LEFTFIRE"]
