import numpy as np
from collections import defaultdict

from classes.helper import GameState, Constants


def same_obj_by_dist(objs, xy2):
    return min([abs(o.xy[0] - xy2[0]) + abs(o.xy[1] - xy2[1])
                for o in objs]) < Constants.MAX_ABS_VELOCITY


def get_id_of_same_obj_by_dist(objs, xy2):
    idx = np.argmin(
        [abs(o.xy[0] - xy2[0]) + abs(o.xy[1] - xy2[1]) for o in objs])
    return objs[idx].id


def get_obj_of_same_obj_by_dist(objs, xy2):
    idx = np.argmin(
        [abs(o.xy[0] - xy2[0]) + abs(o.xy[1] - xy2[1]) for o in objs])
    return objs[idx]


# def player_history_callback(obj_list, prev_obj_coords_dict):
#     for obj in obj_list:
#         if obj.obj_type != 'player':
#             continue
#         player = obj
#         if player.prev_xy in prev_obj_coords_dict[player.obj_type]:
#             old_player = prev_obj_coords_dict[player.obj_type][player.prev_xy]
#             player.history = {
#                 'velocity_x': old_player.history['velocity_x'][-Constants.HISTORY_LENGTH:] + [player.velocity_x],
#                 'velocity_y': old_player.history['velocity_y'][-Constants.HISTORY_LENGTH:] + [player.velocity_y],
#                 'deleted': old_player.history['deleted'][-Constants.HISTORY_LENGTH:] + [player.deleted],
#                 'interactions':  old_player.history['interactions'][-Constants.HISTORY_LENGTH:] + \
#                     [[xx.obj2.obj_type if xx.obj1.obj_type == 'player' else  xx.obj1.obj_type for xx in obj_list.get_player_interactions()]],
#                 'touch_below': old_player.history['touch_below'][-Constants.HISTORY_LENGTH:] + [sum([old_player.touches(other_obj, 3, 0.5) for other_obj in obj_list]) > 0],
#                 'w_change': old_player.history['w_change'][-Constants.HISTORY_LENGTH:] + [player.w_change],
#                 'h_change': old_player.history['h_change'][-Constants.HISTORY_LENGTH:] + [player.h_change],
#             }
#         else:
#             player.history = {
#                 'velocity_x': [player.velocity_x],
#                 'velocity_y': [player.velocity_y],
#                 'deleted': [1,
#                             player.deleted],  # Assume it was deleted earlier
#                 'interactions': [[
#                     xx.obj2.obj_type
#                     if xx.obj1.obj_type == 'player' else xx.obj1.obj_type
#                     for xx in obj_list.get_player_interactions()
#                 ]],
#                 'touch_below': [
#                     sum([
#                         player.touches(other_obj, 3, 0.5)
#                         for other_obj in obj_list
#                     ]) > 0
#                 ],
#                 'w_change': [player.w_change],
#                 'h_change': [player.h_change],
#             }

# def other_obj_visibility_history_callback(obj_list, prev_obj_coords_dict):
#     for obj in obj_list:
#         if obj.obj_type == 'player':
#             continue
#         if obj.prev_xy in prev_obj_coords_dict[obj.obj_type]:
#             old_obj = prev_obj_coords_dict[obj.obj_type][obj.prev_xy]
#             obj.history = {
#                 'velocity_x':
#                 old_obj.history['velocity_x'][-Constants.HISTORY_LENGTH:] +
#                 [obj.velocity_x],
#                 'velocity_y':
#                 old_obj.history['velocity_y'][-Constants.HISTORY_LENGTH:] +
#                 [obj.velocity_y],
#                 'deleted':
#                 old_obj.history['deleted'][-Constants.HISTORY_LENGTH:] +
#                 [obj.deleted],
#                 'w_change':
#                 old_obj.history['w_change'][-Constants.HISTORY_LENGTH:] +
#                 [obj.w_change],
#                 'h_change':
#                 old_obj.history['h_change'][-Constants.HISTORY_LENGTH:] +
#                 [obj.h_change],

#             }
#         else:
#             obj.history = {
#                 'velocity_x': [obj.velocity_x],
#                 'velocity_y': [obj.velocity_y],
#                 'deleted': [1, obj.deleted],  # Assume it was deleted earlier
#                 'w_change': [obj.w_change],
#                 'h_change': [obj.h_change],
#             }

# def player_and_obj_visibility_history_callback(obj_list, prev_obj_coords_dict):
#     player_history_callback(obj_list, prev_obj_coords_dict)
#     other_obj_visibility_history_callback(obj_list, prev_obj_coords_dict)


def update_history_of_new_nonplayer(old_obj, obj):
    if old_obj is None:
        obj.history = {
            'velocity_x': [obj.velocity_x],
            'velocity_y': [obj.velocity_y],
            'deleted': [1, obj.deleted],  # Assume it was deleted earlier
            'w_change': [obj.w_change],
            'h_change': [obj.h_change],
        }
    else:
        obj.history = {
            'velocity_x':
            old_obj.history['velocity_x'][-Constants.HISTORY_LENGTH:] +
            [obj.velocity_x],
            'velocity_y':
            old_obj.history['velocity_y'][-Constants.HISTORY_LENGTH:] +
            [obj.velocity_y],
            'deleted':
            old_obj.history['deleted'][-Constants.HISTORY_LENGTH:] +
            [obj.deleted],
            'w_change':
            old_obj.history['w_change'][-Constants.HISTORY_LENGTH:] +
            [obj.w_change],
            'h_change':
            old_obj.history['h_change'][-Constants.HISTORY_LENGTH:] +
            [obj.h_change],
        }


def update_history_of_new_player(old_player, player, obj_list):
    if old_player is None:
        player.history = {
            'velocity_x': [player.velocity_x],
            'velocity_y': [player.velocity_y],
            'deleted': [1, player.deleted],  # Assume it was deleted earlier
            'interactions': [[
                xx.obj2.obj_type
                if xx.obj1.obj_type == 'player' else xx.obj1.obj_type
                for xx in obj_list.get_player_interactions()
            ]],
            'touch_below': [
                sum([
                    player.touches(other_obj, 3, 0.5) for other_obj in obj_list
                ]) > 0
            ],
            'n_touch_left': [
                sum([
                    player.touches(other_obj, 0, 0.5) for other_obj in obj_list
                ])
            ],
            'n_touch_right': [
                sum([
                    player.touches(other_obj, 1, 0.5) for other_obj in obj_list
                ])
            ],
            'n_touch_above': [
                sum([
                    player.touches(other_obj, 2, 0.5) for other_obj in obj_list
                ])
            ],
            'n_touch_below': [
                sum([
                    player.touches(other_obj, 3, 0.5) for other_obj in obj_list
                ])
            ],
            'w_change': [player.w_change],
            'h_change': [player.h_change],
        }
    else:
        player.history = {
            'velocity_x': old_player.history['velocity_x'][-Constants.HISTORY_LENGTH:] + [player.velocity_x],
            'velocity_y': old_player.history['velocity_y'][-Constants.HISTORY_LENGTH:] + [player.velocity_y],
            'deleted': old_player.history['deleted'][-Constants.HISTORY_LENGTH:] + [player.deleted],
            'interactions':  old_player.history['interactions'][-Constants.HISTORY_LENGTH:] + \
                [[xx.obj2.obj_type if xx.obj1.obj_type == 'player' else  xx.obj1.obj_type for xx in obj_list.get_player_interactions()]],

            'touch_below': old_player.history['touch_below'][-Constants.HISTORY_LENGTH:] + [sum([player.touches(other_obj, 3, 0.5) for other_obj in obj_list]) > 0],

            'n_touch_left': old_player.history['n_touch_left'][-Constants.HISTORY_LENGTH:] + [sum([player.touches(other_obj, 0, 0.5) for other_obj in obj_list])],
            'n_touch_right': old_player.history['n_touch_right'][-Constants.HISTORY_LENGTH:] + [sum([player.touches(other_obj, 1, 0.5) for other_obj in obj_list])],
            'n_touch_above': old_player.history['n_touch_above'][-Constants.HISTORY_LENGTH:] + [sum([player.touches(other_obj, 2, 0.5) for other_obj in obj_list])],
            'n_touch_below': old_player.history['n_touch_below'][-Constants.HISTORY_LENGTH:] + [sum([player.touches(other_obj, 3, 0.5) for other_obj in obj_list])],

            'w_change': old_player.history['w_change'][-Constants.HISTORY_LENGTH:] + [player.w_change],
            'h_change': old_player.history['h_change'][-Constants.HISTORY_LENGTH:] + [player.h_change],
        }


def update_history_of_new_obj(old_obj, obj, obj_list):
    if obj.obj_type == 'player':
        update_history_of_new_player(old_obj, obj, obj_list)
    else:
        update_history_of_new_nonplayer(old_obj, obj)


class ObjectTracker:
    """
    Help track object ids across frames
    """
    def __init__(self, init_obj_list=None):
        self.prev_obj_coords_dict = defaultdict(dict)
        self.prev_objs_by_type = defaultdict(list)
        self.max_obj_id = 0
        if init_obj_list is not None:
            for o in init_obj_list:
                self.prev_obj_coords_dict[o.obj_type][o.xy] = o
                self.prev_objs_by_type[o.obj_type].append(o)

    def reset(self):
        self.prev_obj_coords_dict = defaultdict(dict)
        self.prev_objs_by_type = defaultdict(list)
        self.max_obj_id = 0

    def update(self, obj_list, handle_same_id=False):
        # If new portals are introduced, then everything is new!
        new_room = False
        for o in obj_list:
            # If there is a new portal, then everything is new
            if o.obj_type.startswith('portal') and len(
                    self.prev_objs_by_type[o.obj_type]) == 0:
                new_room = True
                break

        # Assumption: when objects are 'teleported', they are no longer the same object, hence different ids
        for o in obj_list:
            matched_old_obj = None
            if not new_room:
                if o.prev_xy in self.prev_obj_coords_dict[o.obj_type]:
                    matched_old_obj = self.prev_obj_coords_dict[o.obj_type][
                        o.prev_xy]
                elif o.obj_type in self.prev_objs_by_type and same_obj_by_dist(
                        self.prev_objs_by_type[o.obj_type], o.xy):
                    matched_old_obj = get_obj_of_same_obj_by_dist(
                        self.prev_objs_by_type[o.obj_type], o.xy)

            if matched_old_obj is not None:
                o.id = matched_old_obj.id
                o.w_change = o.w - matched_old_obj.w
                o.h_change = o.h - matched_old_obj.h
            else:
                o.id = self.max_obj_id
                # Transfer x and y before setting velocity to 0.
                o.prev_x = o.x
                o.prev_y = o.y
                o.velocity_x = 0
                o.velocity_y = 0
                self.max_obj_id += 1
            update_history_of_new_obj(matched_old_obj, o, obj_list)
            
        if handle_same_id:
            for idx1, o1 in enumerate(obj_list):
                for idx2, o2 in enumerate(obj_list):
                    if idx1 == idx2:
                        continue
                    if o1.id == o2.id and o2.prev_xy in self.prev_obj_coords_dict[o2.obj_type]:
                        o1.id = self.max_obj_id
                        # Transfer x and y before setting velocity to 0.
                        o1.prev_x = o1.x
                        o1.prev_y = o1.y
                        o1.velocity_x = 0
                        o1.velocity_y = 0
                        self.max_obj_id += 1
                        break

        # if self.history_callback is not None:
        #     self.history_callback(obj_list, self.prev_obj_coords_dict)

        self.prev_obj_coords_dict = defaultdict(dict)
        self.prev_objs_by_type = defaultdict(list)
        for o in obj_list:
            self.prev_obj_coords_dict[o.obj_type][o.xy] = o
            self.prev_objs_by_type[o.obj_type].append(o)

    def handle_game_state(self, obj_list, game_state):
        if game_state == GameState.RESTART:
            for o in obj_list:
                if o.obj_type in ['player', 'skull', 'spider']:
                    o.id = self.max_obj_id
                    o.velocity_x = 0
                    o.velocity_y = 0
                    self.max_obj_id += 1
