from classes.game_utils.base_game import BaseActions

pitfall_wh_dict = {
    'player': (8, 21),
    'wall': (7, 32),
    'logs': (6, 14),
    'stairpit': (8, 6),
    'stair': (4, 42),
    'pit': (12, 6),
    'scorpion': (7, 10),
    'rope': (5, 5),
    'snake': (8, 14),
    'tarpit': (64, 10),
    'waterhole': (64, 10),
    'crocodile': (8, 8),
    'goldenbar': (7, 13),
    'silverbar': (7, 13),
    'diamondring': (7, 13),
    'fire': (8, 14),
    'moneybag': (7, 14),
    'platform': (8, 4),
    'lifecount': (1, 8),
    'playerscore': (6, 8),
    'timer': (37, 8),
}

# Extra
for i in range(20):
    pitfall_wh_dict[f'portal_{i}'] = (1, 40)
pitfall_wh_dict['movinglogs'] = (6, 14)
pitfall_wh_dict['disappearingtarpit'] = (0, 0)
pitfall_wh_dict['closedcrocodile'] = (8, 6)
pitfall_wh_dict['opencrocodile'] = (8, 9)
pitfall_wh_dict['platform'] = (152, 1)
pitfall_wh_dict['dangerousplatform'] = (152, 1)

PITFALL_MAX_ABS_VELOCITY = 15

PITFALL_HISTORY_LENGTH = 100

PITFALL_MAX_ABS_SIZE_CHANGE = 30


class PitfallActions(BaseActions):
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
