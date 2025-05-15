from classes.game_utils.base_game import BaseActions

montezuma_revenge_wh_dict = {
    'player': (8, 20),
    'skull': (7, 13),
    'spider': (8, 11),
    'snake': (7, 13),
    'key': (7, 15),
    'amulet': (6, 15),
    'torch': (6, 13),
    'sword': (6, 15),
    'ruby': (7, 12),
    'barrier': (4, 37),
    'beam': (4, 40),
    'rope': (1, 39),
    'score': (5, 8),
    'life': (7, 5),
    'key_hud': (7, 15),
    'amulet_hud': (6, 15),
    'torch_hud': (6, 13),
    'sword_hud': (6, 15),
    'platform': (8, 4),
    'ladder': (8, 4),
    'conveyer_belt': (8, 4),
    'wall': (8, 4),
    'disappearing_platform': (8, 4)
}

MONTEZUMA_REVENGE_MAX_ABS_VELOCITY = 15

MONTEZUMA_REVENGE_HISTORY_LENGTH = 100

MONTEZUMA_REVENGE_MAX_ABS_SIZE_CHANGE = 1

MONTEZUMA_REVENGE_ACTIONS = ['NOOP', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'FIRE', 'LEFTFIRE', 'RIGHTFIRE']


class MontezumaRevengeActions(BaseActions):
    NOOP = 'NOOP'
    FIRE = 'FIRE'
    UP = 'UP'
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    DOWN = 'DOWN'
    UPRIGHT = 'UPRIGHT'
    UPLEFT = 'UPLEFT'
    DOWNRIGHT = 'DOWNRIGHT'
    DOWNLEFT = 'DOWNLEFT'
    UPFIRE = 'UPFIRE'
    RIGHTFIRE = 'RIGHTFIRE'
    LEFTFIRE = 'LEFTFIRE'
    DOWNFIRE = 'DOWNFIRE'
    UPRIGHTFIRE = 'UPRIGHTFIRE'
    UPLEFTFIRE = 'UPLEFTFIRE'
    DOWNRIGHTFIRE = 'DOWNRIGHTFIRE'
    DOWNLEFTFIRE = 'DOWNLEFTFIRE'

    def get_all_possible_actions():
        return [
            'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
            'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE',
            'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE',
            'DOWNLEFTFIRE'
        ]
