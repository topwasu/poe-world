#!/usr/bin/env python
# coding=utf-8

DocString = '''
Here are some documentation explaining the API for the environment.
"""
class StateTransitionTriplet:
    Attributes:
        input_state (ObjList): list of objects in the input state
        event (str): action taken in the input state
        output_state (ObjList): list of objects in the output state
class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added
class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int): x-axis velocity of the object
        velocity_y (int): y-axis velocity of the object

    Methods:
        touches(obj: Obj, touch_side: int, touch_percent: float) -> bool:
            Returns whether this Obj is touching the input obj (True/False)
            based on the input touch_side (0 = left, 1 = right, 2 = up, 3 = down) and touch_percent (threshold for touching area percentage)
"""
'''

TransitCodeExample = '''
Here is an example of the transition function that you could implement.
```
def transition(state, event):
    """
    Args:
        state (ObjList): list of objects in the input state
        event (str): action taken in the input state
    Returns:
        new_state (ObjList): list of objects in the output state
    """
    # Example of transition, just showing how to access objects and modify them
    new_state = state
    player = state.get_objs_by_obj_type("player")[0]
    ball = state.get_objs_by_obj_type("ball")[0]
    if player.touches(ball, 0, 1.0):
        ball.velocity_x = -ball.velocity_x
    if event == 'UP':
        player.velocity_y = player.velocity_y + 1
    return new_state
```
'''
