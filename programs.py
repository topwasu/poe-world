from classes.helper import *

p1 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action in ['LEFT', 'RIGHT']:
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.center_y = RandomValues([platform_obj.center_y - 11])
                    break
    return obj_list
"""

p2 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'DOWN':
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.center_x = RandomValues([ladder_obj.center_x])
                    break
    return obj_list
"""

p3 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'LEFT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([-3])
                    # player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p4 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'LEFT':
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p5 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'DOWN':
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_y = RandomValues([3, 11])
                    break
    return obj_list
"""

p6 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'DOWN':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p7 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'NOOP':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p8 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'NOOP':
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p9 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action == 'UP':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([0])
                    break
    return obj_list
"""

p10 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=1, shrink_side=0) -> ObjList:
    if action == 'UP':
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.velocity_x = RandomValues([0])
                    player_obj.velocity_y = RandomValues([-2, -3])
                    break
    return obj_list
"""

p11 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action in ['LEFT', 'RIGHT']:
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, shrink_top, shrink_bottom, shrink_side):
                    print(player_obj.center_y, platform_obj.center_y - 11)
                    player_obj.center_y = RandomValues([platform_obj.center_y - 11])
                    print(player_obj.velocity_y)
                    break
    return obj_list
"""

p12 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action in ['DOWN', 'UP']:
        player_objs = obj_list.get_objs_by_obj_type('player')
        ladder_objs = obj_list.get_objs_by_obj_type('ladder')
        
        for player_obj in player_objs:
            for ladder_obj in ladder_objs:
                if player_obj.touches(ladder_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.center_x = RandomValues([ladder_obj.center_x])
                    break
    return obj_list
"""

p13 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    if action in ['LEFT', 'RIGHT']:
        player_objs = obj_list.get_objs_by_obj_type('player')
        conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')
        
        for player_obj in player_objs:
            for conveyer_belt_obj in conveyer_belt_objs:
                if player_obj.touches(conveyer_belt_obj, shrink_top, shrink_bottom, shrink_side):
                    player_obj.center_y = RandomValues([conveyer_belt_obj.center_y - 11])
                    break
    return obj_list
"""

p14 = """\
def alter_player_objects(obj_list: ObjList, action: str, shrink_top=0, shrink_bottom=0, shrink_side=0) -> ObjList:
    player_objs = obj_list.get_objs_by_obj_type('player')
    rope_objs = obj_list.get_objs_by_obj_type('rope')
    
    for player_obj in player_objs:
        for rope_obj in rope_objs:
            if player_obj.touches(rope_obj, shrink_top, shrink_bottom, shrink_side):
                player_obj.center_x = RandomValues([rope_obj.center_x])
                break
    return obj_list
"""

new_c0 = """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.3) -> bool:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == platform_obj.top_side:
                    n_satisfied += 1
                    break  # No need to check other platforms if this one satisfies the condition
    
    return n_touch, n_satisfied"""

new_c1 = """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.3) -> bool:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == ladder_obj.center_x:
                    n_satisfied += 1
                    break  # No need to check other ladders if constraint is satisfied for this player
        
    return n_touch, n_satisfied"""

new_c2 = """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.3) -> bool:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for conveyer_belt_obj in conveyer_belt_objs:  # conveyer_belt_obj is of type Obj
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    n_satisfied += 1
                    break  # No need to check other conveyer belts for this player
    
    return n_touch, n_satisfied"""

newnew_c0 = """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == platform_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied"""

newnew_c1 = """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')
    
    for player_obj in player_objs:
        for ladder_obj in ladder_objs:
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == ladder_obj.center_x:
                    n_satisfied += 1
    
    return n_touch, n_satisfied"""

newnew_c2 = """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')
    
    for player_obj in player_objs:
        for conveyer_belt_obj in conveyer_belt_objs:
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied"""

new_constraints = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')
    
    for player_obj in player_objs:
        for ladder_obj in ladder_objs:
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == ladder_obj.center_x:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == platform_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')
    
    for player_obj in player_objs:
        for conveyer_belt_obj in conveyer_belt_objs:
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')
    
    for player_obj in player_objs:
        for ladder_obj in ladder_objs:
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.left_side == ladder_obj.right_side:
                    n_satisfied += 1
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.30000000000000004) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == rope_obj.center_x:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.left_side == rope_obj.right_side:
                    n_satisfied += 1
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=1, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.right_side == rope_obj.left_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
]

new_constraints_prune_hard = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')
    
    for player_obj in player_objs:
        for ladder_obj in ladder_objs:
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == ladder_obj.center_x:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.0) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == platform_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.7000000000000001) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')
    
    for player_obj in player_objs:
        for conveyer_belt_obj in conveyer_belt_objs:
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.30000000000000004) -> tuple:
    n_touch, n_satisfied = 0, 0
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                n_touch += 1
                if player_obj.center_x == rope_obj.center_x:
                    n_satisfied += 1
    
    return n_touch, n_satisfied""",
]

test_constraints = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> tuple:
    n_touch, n_satisfied = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')
    
    for player_obj in player_objs:
        for ladder_obj in ladder_objs:
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                n_touch.append(ladder_obj.id)
                if player_obj.center_x == ladder_obj.center_x:
                    n_satisfied.append(ladder_obj.id)
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.3) -> tuple:
    n_touch, n_satisfied = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                n_touch.append(platform_obj.id)
                if player_obj.bottom_side == platform_obj.top_side:
                    n_satisfied.append(platform_obj.id)
    
    return n_touch, n_satisfied""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.7000000000000001) -> tuple:
    n_touch, n_satisfied = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')
    
    for player_obj in player_objs:
        for conveyer_belt_obj in conveyer_belt_objs:
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                n_touch.append(conveyer_belt_obj.id)
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    n_satisfied.append(conveyer_belt_obj.id)
    
    return n_touch, n_satisfied""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.30000000000000004) -> tuple:
    n_touch, n_satisfied = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                n_touch.append(rope_obj.id)
                if player_obj.center_x == rope_obj.center_x:
                    n_satisfied.append(rope_obj.id)
    
    return n_touch, n_satisfied""",
]

test_constraints2 = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=2, touch_percent=0.30000000000000004) -> tuple:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                touch_ids.append(player_obj.id)
                if player_obj.center_x == rope_obj.center_x:
                    satisfied_ids.append(player_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=0, touch_percent=0.8) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                touch_ids.append(ladder_obj.id)
                if player_obj.left_side == ladder_obj.right_side:
                    satisfied_ids.append(ladder_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.1) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for conveyer_belt_obj in conveyer_belt_objs:  # conveyer_belt_obj is of type Obj
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                touch_ids.append(conveyer_belt_obj.id)
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    satisfied_ids.append(conveyer_belt_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                touch_ids.append(platform_obj.id)
                if player_obj.bottom_side == platform_obj.top_side:
                    satisfied_ids.append(platform_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                touch_ids.append(ladder_obj.id)
                if player_obj.center_x == ladder_obj.center_x:
                    satisfied_ids.append(ladder_obj.id)
    
    return touch_ids, satisfied_ids""",
]

test_constraints2_wo_left = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=2, touch_percent=0.30000000000000004) -> tuple:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                touch_ids.append(player_obj.id)
                if player_obj.center_x == rope_obj.center_x:
                    satisfied_ids.append(player_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.1) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for conveyer_belt_obj in conveyer_belt_objs:  # conveyer_belt_obj is of type Obj
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                touch_ids.append(conveyer_belt_obj.id)
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    satisfied_ids.append(conveyer_belt_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                touch_ids.append(platform_obj.id)
                if player_obj.bottom_side == platform_obj.top_side:
                    satisfied_ids.append(platform_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                touch_ids.append(ladder_obj.id)
                if player_obj.center_x == ladder_obj.center_x:
                    satisfied_ids.append(ladder_obj.id)
    
    return touch_ids, satisfied_ids""",
]

test_constraints3 = [
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=2, touch_percent=0.30000000000000004) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    rope_objs = obj_list.get_objs_by_obj_type('rope')  # get all Obj of type 'rope'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for rope_obj in rope_objs:  # rope_obj is of type Obj
            if player_obj.touches(rope_obj, touch_side, touch_percent):
                touch_ids.append(rope_obj.id)
                if player_obj.center_x == rope_obj.center_x:
                    satisfied_ids.append(rope_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=0, touch_percent=0.8) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                touch_ids.append(ladder_obj.id)
                if player_obj.left_side == ladder_obj.right_side:
                    satisfied_ids.append(ladder_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.1) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for conveyer_belt_obj in conveyer_belt_objs:  # conveyer_belt_obj is of type Obj
            if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                touch_ids.append(conveyer_belt_obj.id)
                if player_obj.bottom_side == conveyer_belt_obj.top_side:
                    satisfied_ids.append(conveyer_belt_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_y_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=0.5) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for platform_obj in platform_objs:  # platform_obj is of type Obj
            if player_obj.touches(platform_obj, touch_side, touch_percent):
                touch_ids.append(platform_obj.id)
                if player_obj.bottom_side == platform_obj.top_side:
                    satisfied_ids.append(platform_obj.id)
    
    return touch_ids, satisfied_ids""",
    """def check_x_of_player_objects(obj_list: ObjList, _, touch_side=3, touch_percent=1.0) -> ObjList:
    touch_ids, satisfied_ids = [], []
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    ladder_objs = obj_list.get_objs_by_obj_type('ladder')  # get all Obj of type 'ladder'
    
    for player_obj in player_objs:  # player_obj is of type Obj
        for ladder_obj in ladder_objs:  # ladder_obj is of type Obj
            if player_obj.touches(ladder_obj, touch_side, touch_percent):
                touch_ids.append(ladder_obj.id)
                if player_obj.center_x == ladder_obj.center_x:
                    satisfied_ids.append(ladder_obj.id)
    
    return touch_ids, satisfied_ids""",
]

txt = """\
def get_player_objects(obj_list: ObjList, touch_side=3, touch_percent=1.0) -> bool:
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    platform_objs = obj_list.get_objs_by_obj_type('platform')  # get all Obj of type 'platform'
    res = False
    for player_obj in player_objs:  # player_obj is of type Obj
        if player_obj.falling_time >= 5:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, touch_side, touch_percent):
                    res = True
                    break
    return res
AND
def alter_player_objects(obj_list: ObjList, _, touch_side=-1, touch_percent=0.1) -> ObjList:
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    for player_obj in player_objs:  # player_obj is of type Obj
        # Set the deleted attribute to 1 using RandomValues
        player_obj.deleted = RandomValues([1])
    return obj_list"""

txt2 = """\
def get_player_objects(obj_list: ObjList, touch_side=3, touch_percent=1.0) -> bool:
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    conveyer_belt_objs = obj_list.get_objs_by_obj_type('conveyer_belt')  # get all Obj of type 'conveyer_belt'
    res = False
    for player_obj in player_objs:  # player_obj is of type Obj
        if player_obj.falling_time >= 5:
            for conveyer_belt_obj in conveyer_belt_objs:
                if player_obj.touches(conveyer_belt_obj, touch_side, touch_percent):
                    res = True
                    break
    return res
AND
def alter_player_objects(obj_list: ObjList, _, touch_side=-1, touch_percent=0.1) -> ObjList:
    player_objs = obj_list.get_objs_by_obj_type('player')  # get all Obj of type 'player'
    for player_obj in player_objs:  # player_obj is of type Obj
        # Set the deleted attribute to 1 using RandomValues
        player_obj.deleted = RandomValues([1])
    return obj_list"""

pomdp_test_program = """\
def alter_player_objects(obj_list: ObjList, action: str, touch_side=3, touch_percent=0.1) -> ObjList:
    if action == 'RIGHTFIRE':
        player_objs = obj_list.get_objs_by_obj_type('player')
        platform_objs = obj_list.get_objs_by_obj_type('platform')
        print(player_objs[0])
        for player_obj in player_objs:
            for platform_obj in platform_objs:
                if player_obj.touches(platform_obj, touch_side, touch_percent):
                    player_obj.velocity_y = SeqValues([-6, -7, -4, 0, 2, 6, 9])
                    break
    return obj_list
"""

pomdp2_programs = [
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        # Becoming visible means the object is visible even though it was invisible before
        condition = beam_obj.history['deleted'][-1] == 0 and beam_obj.history['deleted'][-2] == 1
        if condition:
            # Set the 'deleted' attribute to a sequence indicating visibility
            beam_obj.deleted = SeqValues([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return obj_list
""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        beam_obj.velocity_x = RandomValues([0])
    return obj_list
""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        beam_obj.velocity_y = RandomValues([0])
    return obj_list
""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        beam_obj.deleted = RandomValues([1])
    return obj_list
""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        # Check if the beam object is disappearing
        condition = beam_obj.history['deleted'][-1] == 1 and beam_obj.history['deleted'][-2] == 0
        if condition:
            # Create new beam objects at specified positions after 13 timesteps
            new_positions = [(112, 53), (120, 53), (140, 53), (16, 53), (36, 53), (44, 53)]
            for x, y in new_positions:
                obj_list = obj_list.create_object('beam', x, y, delay_timesteps=12)
            break  # Avoid setting each attribute value for each beam object more than once
    return obj_list""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of type 'beam'
    if not beam_objs:  # if there are no beam objects
        # Create new beam objects at specified coordinates
        obj_list = obj_list.create_object('beam', 140, 53)
        obj_list = obj_list.create_object('beam', 120, 53)
        obj_list = obj_list.create_object('beam', 112, 53)
        obj_list = obj_list.create_object('beam', 44, 53)
        obj_list = obj_list.create_object('beam', 36, 53)
        obj_list = obj_list.create_object('beam', 16, 53)
    return obj_list""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of type 'beam'
    if not beam_objs:
        # Create new beam objects at specified positions after 13 timesteps
        new_positions = [(112, 53), (120, 53), (140, 53), (16, 53), (36, 53), (44, 53)]
        for x, y in new_positions:
            obj_list = obj_list.create_object('beam', x, y, delay_timesteps=12)
    return obj_list""",
    """\
def alter_beam_objects(obj_list: ObjList, _, touch_side=0, touch_percent=1.0) -> ObjList:
    beam_objs = obj_list.get_objs_by_obj_type('beam')  # get all Obj of obj_type 'beam'
    for beam_obj in beam_objs:  # beam_obj is of type Obj
        # Check if deleted suffix has been 1 
        condition = beam_obj.history['deleted'][-1] == 1 and beam_obj.history['deleted'][-2] == 0
        if condition:
            # Create new beam objects at specified positions after 13 timesteps
            new_positions = [(112, 53), (120, 53), (140, 53), (16, 53), (36, 53), (44, 53)]
            for x, y in new_positions:
                obj_list = obj_list.create_object('beam', x, y, delay_timesteps=12)
            break  # Avoid setting each attribute value for each beam object more than once
    return obj_list""",
]
