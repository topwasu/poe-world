from classes.game_utils.base_game import BaseActions

breakout_wh_dict = {
    'player': (16, 4),
    'ball': (2, 4),
    'block': (8, 6),
    'playerscore': (44, 10),
    'live': (12, 10),
    'playernumber': (4, 10)
}

BREAKOUT_MAX_ABS_VELOCITY = 15

BREAKOUT_HISTORY_LENGTH = 100

BREAKOUT_MAX_ABS_SIZE_CHANGE = 1


class BreakoutActions(BaseActions):
    NOOP = 'NOOP'
    FIRE = 'FIRE'
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'

    def get_all_possible_actions():
        return ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
