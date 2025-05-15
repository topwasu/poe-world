propose_prompt = \
"""\
"""

explain_event_prompt = """\
We observe that the possible effects of {action} on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of obj_type '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of action '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are conflicting changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
6. Please use if-condition to indicate that the effects only happen because of action '{action}'
Format the output as a numbered list.
"""

explain_event_pomdp_prompt = """\
We observe that the possible effects of {action} on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of obj_type '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class SeqValues:
    Use this class to express a sequence of values.

    Attributes:
        sequence (list[ints]): list of possible values

    Methods:
        __init__(sequence):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | SeqValues): x-axis velocity of the object
        velocity_y (int | SeqValues): y-axis velocity of the object
        deleted (int | SeqValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of action '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use SeqValues to set attribute values. Construct it with the full list of values, even if all elements in the list are the same.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
6. Please use if-condition to indicate that the effects only happen because of action '{action}'
Format the output as a numbered list.
"""

explain_status_event_pomdp_prompt = """\
We observe that the possible effects of '{action}' on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of obj_type '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # Becoming means the object is visible even though it was invisible before
        # condition = {obj_type}_obj.history['deleted'][-1] == 0 and {obj_type}_obj.history['deleted'][-2] == 1
        # while disappearing means the object is invisible even though it was visible before
        # condition = {obj_type}_obj.history['deleted'][-1] == 1 and {obj_type}_obj.history['deleted'][-2] == 0:
        if condition:
            pass
    return obj_list

And here are the docstrings for relevant classes:

class SeqValues:
    Use this class to express a sequence of values.

    Attributes:
        sequence (list[ints]): list of possible values

    Methods:
        __init__(sequence):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | SeqValues): x-axis velocity of the object
        velocity_y (int | SeqValues): y-axis velocity of the object
        deleted (int | SeqValues): indicates visibility / whether this object gets deleted (1 if it does and 0 if it does not)
        history (dict): history of the object (e.g. history['deleted'] = [0, 1, 1, 1, 1])

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int, delay_timesteps: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added, with its history['deleted'] being set to 1 for (delay_timesteps-1) timesteps before being set to 0

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the deleted / visibility or creation of object.
2. Always use SeqValues to set attribute values. Construct it with the full list of values, even if all elements in the list are the same.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. Please use if-condition to indicate that the effects only happen because of '{action}'
6. Function names all have to be alter_{obj_type}_objects
Format the output as a numbered list.
"""

explain_status_event_pomdp_2_prompt = """\
We observe that the possible effects of '{action}' on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of obj_type '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # Becoming means the object is visible even though it was invisible before
        # condition = {obj_type}_obj.history['deleted'][-1] == 0 and {obj_type}_obj.history['deleted'][-2] == 1
        # while disappearing means the object is invisible even though it was visible before
        # condition = {obj_type}_obj.history['deleted'][-1] == 1 and {obj_type}_obj.history['deleted'][-2] == 0:
        if condition:
            pass
    return obj_list

And here are the docstrings for relevant classes:

class SeqValues:
    Use this class to express a sequence of values.

    Attributes:
        sequence (list[ints]): list of possible values

    Methods:
        __init__(sequence):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        w_change (int | SeqValues): width change rate of the object
        h_change (int | SeqValues): height change rate of the object
        deleted (int | SeqValues): indicates visibility / whether this object gets deleted (1 if it does and 0 if it does not)
        history (dict): history of the object (e.g. history['deleted'] = [0, 1, 1, 1, 1])

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int, delay_timesteps: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added, with its history['deleted'] being set to 1 for (delay_timesteps-1) timesteps before being set to 0

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the deleted / visibility or creation of object.
2. Always use SeqValues to set attribute values. Construct it with the full list of values, even if all elements in the list are the same.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. Please use if-condition to indicate that the effects only happen because of '{action}'
6. Function names all have to be alter_{obj_type}_objects
Format the output as a numbered list.
"""

explain_event_symmetric_prompt = """\
Given this list of actions: NOOP, UP, DOWN, RIGHT, LEFT, FIRE, LEFTFIRE, RIGHTFIRE

Some of these actions are the opposite of each other.
For example, the opposite of LEFT is RIGHT, the opposite action of UP is DOWN, and the opposite of LEFTFIRE is RIGHTFIRE.
Some actions do not have an opposite, such as NOOP and FIRE. 

You will be given a python function with the following signature: 

def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList

This function describes the possible effects of {action} on {obj_type} objects.
Please output a function with the same signature that describes what happens when the 'opposite' action is taken.
If {action} has no opposite, instead of outputting a function, just output the text 'NONE'

Here is the input python function:
{func}
"""

explain_event_passive_prompt = """\
We observe that the possible behaviors of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible behaviors of {obj_type} objects following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are multiple possible changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches_where to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
Format the output as a numbered list.
"""

explain_event_v_tracking_prompt = """\
We observe that the possible effects of {action} on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        new_velocity_x (int): new x-axis velocity of the object
        new_velocity_y (int): new y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible effects of action '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are conflicting changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
6. Please use if-condition to indicate that the effects only happen because of action '{action}'
Format the output as a numbered list.
"""

explain_event_passive_pomdp_prompt = """\
We observe that the possible behaviors of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class SeqValues:
    Use this class to express a sequence of values.

    Attributes:
        sequence (list[ints]): list of possible values

    Methods:
        __init__(sequence):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        x (int): x-axis position of the object
        y (int): y-axis position of the object
        velocity_x (int | SeqValues): x-axis velocity of the object
        velocity_y (int | SeqValues): y-axis velocity of the object
        deleted (int | SeqValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible behaviors of {obj_type} objects following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use SeqValues to set attribute values. Construct it with the full list of values, even if all elements in the list are the same.
3. Use Obj.touches_where to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
Format the output as a numbered list.
"""

explain_event_passive_pomdp_2_prompt = """\
We observe that the possible behaviors of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class SeqValues:
    Use this class to express a sequence of values.

    Attributes:
        sequence (list[ints]): list of possible values

    Methods:
        __init__(sequence):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        w (int): width of the object
        h (int): height of the object
        w_change (int | SeqValues): width change rate of the object
        h_change (int | SeqValues): height change rate of the object

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible behaviors of {obj_type} objects following these rules:
1. Each function should make changes to one attribute -- this could be width change rate or height change rate of the object.
2. Always use SeqValues to set attribute values. Construct it with the full list of values, even if all elements in the list are the same.
3. Use Obj.touches_where to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
Format the output as a numbered list.
"""

explain_event_passive_danger_prompt = """\
We observe that the possible behaviors of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

def compute_danger_attribute(obj: Obj) -> int:
    This function computes the danger attribute of an object based on its type and other attributes.

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains each of the {n} possible behaviors of {obj_type} objects following these rules:
1. Each function should make changes to one attribute -- this could be the creation of object or deletion of object.
2. Always use RandomValues to set attribute values. If there are multiple possible changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches_where to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
Format the output as a numbered list, each with its python code block - so there should be {n} python code blocks.
"""

explain_event_hud_prompt = """\
We observe that the possible effects of existence of hud objects on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of obj_type '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains the possible effects of existence of certain hud objects following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are conflicting changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. You can assume the velocities of input objects are integers.
6. Please use if-condition to indicate that the effects only happen because of existence of certain hud objects
Format the output as a numbered list.
"""

explain_event_hud_item_prompt = """\
We observe that the possible behaviors of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int | RandomValues): x-axis velocity of the object
        velocity_y (int | RandomValues): y-axis velocity of the object
        deleted (int | RandomValues): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis position, y-axis position, creation of object, or deletion of object.
2. Always use RandomValues to set attribute values. If there are multiple possible changes to an attribute, instantiate RandomValues with a list of all possible values for that attribute.
3. Use Obj.touches_where to check for interactions.
4. You can assume the velocities of input objects are integers.
5. Each function only talks about one possible effect.
Format the output as a numbered list.
"""

explain_event_snapping_prompt = """\
We observe that the possible effects of {action} on {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements these effects. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Attributes:
        values (list[ints]): list of possible values

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        center_x (int | RandomValues): x-axis center position of the object
        center_y (int | RandomValues): y-axis center position of the object
        left_side (int | RandomValues): left x-axis position of the object
        right_side (int | RandomValues): right x-axis position of the object
        top_side (int | RandomValues): top y-axis position of the object
        bottom_side (int | RandomValues): bottom y-axis position of the object
        new_center_x (int): new x-axis center position of the object
        new_center_y (int): new y-axis center position of the object
        new_left_side (int): new left x-axis position of the object
        new_right_side (int): new right x-axis position of the object
        new_top_side (int): new top y-axis position of the object
        new_bottom_side (int): new bottom y-axis position of the object

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output {n} different alter_{obj_type}_objects functions that explains the possible effects of action '{action}' following these rules:
1. Each function should make changes to one attribute -- this could be the x-axis center position or y-axis center position.
2. Always use RandomValues to set attribute values.
3. Use Obj.touches_where to check for interactions.
4. Avoid setting each attribute value for each {obj_type} object more than once. For example, use 'break' inside a nested loop.
5. Please use if-condition to indicate that the effects only happen because of action '{action}'.
Format the output as a numbered list.
"""

explain_event_constraints_prompt = """\
We observe that the possible constraints of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that checks whether the x or y position of {obj_type} objects satisfy the constraint. The format of the functions should be

def check_{axis}_of_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    touch_ids, satisfied_ids = [], []
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # if {obj_type}_obj touches the other object mentioned in the constraint, append that other object's id to touch_ids
        # and then if {obj_type}_obj satisfies the constraint touching the other object, append that other object's id to satisfied_ids
        pass
    return touch_ids, satisfied_ids

And here are the docstrings for relevant classes:

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        center_x (int | RandomValues): x-axis center position of the object
        center_y (int | RandomValues): y-axis center position of the object
        left_side (int | RandomValues): left x-axis position of the object
        right_side (int | RandomValues): right x-axis position of the object
        top_side (int | RandomValues): top y-axis position of the object
        bottom_side (int | RandomValues): bottom y-axis position of the object

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

Please output a check_{axis}_of_{obj_type}_objects function following these rules:
1. Use Obj.touches to check for interactions.
Format the output as a python code block (using ```). Do not explain.
"""

translate_prompt = """\
We observe that a possible behavior of {obj_type} objects include
{obs_lst_txt}

We want to synthesize python functions that implements the effect. The format of the functions should be

def alter_{obj_type}_objects(obj_list: ObjList, _) -> ObjList:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        # You can assume {obj_type}_obj.velocity_x and {obj_type}_obj.velocity_y are integers.
        pass
    return obj_list

And here are the docstrings for relevant classes:

class RandomValues:
    Use this class to express the possibility of random values. Example x = RandomValues([x + 2, x - 2])

    Methods:
        __init__(values):
            Initialize an instance

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int): x-axis velocity of the object
        velocity_y (int): y-axis velocity of the object
        deleted (int): whether this object gets deleted (1 if it does and 0 if it does not)

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

Please output the alter_{obj_type}_objects function following these rules:
1. Always use RandomValues to set attribute values (BUT not for create_object)
2. Do not attempt to set velocities to newly created object
"""


interpret_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 8 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects with x-axis velocity = +0 set their x-axis velocity to +0
2. The player objects with y-axis velocity = +2 set their y-axis velocity to -4
3. The player objects that touch an unknown object set their x-axis velocity to +0
4. The player objects that touch an ladder object set their x-axis velocity to +0
5. The player objects are not deleted.
6. No player objects are created.

Please output a list of 8 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_obj_interact_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to +0
2. The player objects that touch an unknown object set their y-axis velocity to -4
3. The player objects that touch an ladder object set their x-axis velocity to +0
4. The player objects that touch an ladder object set their y-axis velocity to -4

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_obj_interact_pomdp_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0),
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to [+4, +0, -4]
- The player object (id = 0) sets y-axis velocity to [+0, +0, +0]

Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to [+4, +0, -4]
2. The player objects that touch an unknown object set their y-axis velocity to [+0, +0, +0]
3. The player objects that touch an ladder object set their x-axis velocity to [+4, +0, -4]
4. The player objects that touch an ladder object set their y-axis velocity to [+0, +0, +0]

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_obj_momentum_pomdp_x_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 1 possible reasons for the effects

Here's an example with car objects:
Example input list of objects:
car object (id = 0) with x-axis velocity = -3,
car object (id = 0) is at x=32,

Example output list of object changes:
- The car object (id = 0) sets x-axis velocity to [-4, -2, +0]

Example reasons:
1. The car objects with negative x-axis velocity and x-axis position less than or equal to 32 set their x-axis velocity to [-4, -2, +0]

Please output a list of 1 reason of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. If the input velocity is negative, refer to the position with "less than or equal to". If the input velocity is positive, refer to the position with "greater than or equal to". 
"""

interpret_obj_momentum_pomdp_y_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 1 possible reasons for the effects

Here's an example with car objects:
Example input list of objects:
car object (id = 0) with y-axis velocity = +2,
car object (id = 0) is at y=34,

Example output list of object changes:
- The car object (id = 0) sets y-axis velocity to [+5, +3, +4]

Example reasons:
1. The car objects with positive y-axis velocity and y-axis position more than or equal to 34 set their y-axis velocity to [+5, +3, +4]

Please output a list of 1 reason of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. If the input velocity is negative, refer to the position with "less than or equal to". If the input velocity is positive, refer to the position with "greater than or equal to".
"""

interpret_obj_velocity_pomdp_x_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 2 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +4,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to [+0, +0, +2]

Example reasons:
1. The player objects that touch a ladder object with positive x-axis velocity set their x-axis velocity to [+0, +0, +2]
2. The player objects that touch an unknown object with positive x-axis velocity set their x-axis velocity to [+0, +0, +2]

Please output a list of 2 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
4. Only refer to the input velocities only as positive or negative values. Do not use equal, less than, or more than.
"""

interpret_obj_velocity_pomdp_y_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 2 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with y-axis velocity = +4,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets y-axis velocity to [+0, +0, +2]

Example reasons:
1. The player objects that touch a ladder object with positive y-axis velocity set their y-axis velocity to [+0, +0, +2]
2. The player objects that touch an unknown object with positive y-axis velocity set their y-axis velocity to [+0, +0, +2]

Please output a list of 2 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
4. Only refer to the input velocities only as positive or negative values. Do not use equal, less than, or more than.
"""

# interpret_obj_size_change_pomdp_prompt = \
# """\
# I'll give you an input list of objects and an output list of object changes, and I want you to list 2 possible reasons for the effects

# Here's an example with car objects:
# Example input list of objects:
# car object (id = 0) with width change rate = -3 and height change rate = +2,
# car object (id = 0) has its size equal to (w=32,h=34),

# Example output list of object changes:
# - The car object (id = 0) sets width change rate to [-4, -2, +0]
# - The car object (id = 0) sets height change rate to [+5, +3, +4]

# Example reasons:
# 1. The car objects with negative width change rate and width greater than or equal to 32 set their width change rate to [-4, -2, +0]
# 2. The car objects with positive height change rate and height less than or equal to 34 set their height change rate to [+5, +3, +4]

# Please output a list of 2 reason of the {obj_type} objects for the following input and output list of objects.

# Input list of objects:
# {input}

# Output list of object changes:
# {effects}

# Please follow these rules for your output:
# 1. make sure each reason only talks about one object change
# 2. do not talk about IDs
# 3. If width or height is positive, use "less than or equal to". Otherwise, use "greater than or equal to"
# """

interpret_obj_size_change_pomdp_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 2 possible reasons for the effects

Here's an example with car objects:
Example input list of objects:
car object (id = 0) with width change rate = -3 and height change rate = +2,
car object (id = 0) has its size equal to (w=32,h=34),

Example output list of object changes:
- The car object (id = 0) sets width change rate to [-4, -2, +0]
- The car object (id = 0) sets height change rate to [+5, +3, +4]

Example reasons:
1. The car objects with negative width change rate and width equal to 32 set their width change rate to [-4, -2, +0]
4. The car objects with positive height change rate and height equal to 34 set their height change rate to [+5, +3, +4]

Please output a list of 2 reason of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_obj_change_pomdp_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 6 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0),

Example output list of object changes:
- A new player object is created at (x=32,y=34) after 3 timesteps
- The player object (id = 0) sets its visibility to [+0, +0, +0]

Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to [+4, +0, -4]
2. The player objects that touch an unknown object set their y-axis velocity to [+0, +0, +0]
3. The player objects that touch an ladder object set their x-axis velocity to [+4, +0, -4]
4. The player objects that touch an ladder object set their y-axis velocity to [+0, +0, +0]

Please output a list of 6 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_velocity_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 6 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects that touch an unknown object set their x-axis velocity to +0
2. The player objects that touch an unknown object set their y-axis velocity to -4
3. The player objects that touch an ladder object set their x-axis velocity to +0
4. The player objects that touch an ladder object set their y-axis velocity to -4
5. The player objects are not deleted.
6. No player objects are created.

Please output a list of 6 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
"""

interpret_velocity_creation_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0),
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) is deleted
- A new player object is created at (x=76,y=73)

Example reasons:
1. If player objects touch an unknown object, a new player object is created at (x=76, y=73)
2. If player objects touch a ladder object, a new player object is created at (x=76, y=73)
3. The player objects that touch an unknown object are deleted
4. The player objects that touch a ladder object are deleted

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
"""

interpret_velocity_2_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 8 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
,

Example output list of object changes:
- A new player object is created at (x=96,y=37)
- A new player object is created at (x=32,y=34)
- A new player object is created at (x=64,y=64)


Example reasons:
1. All input player objects are deleted / not visible, new player objects are created at (x=96, y=37), (x=32, y=34), and (x=64, y=64)

Please output a list of 1 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. If objects are created at several positions, put them together in a single sentence. For example, 'If there is no input player objects, new player objects are created at (x=96, y=37), (x=32, y=34), and (x=64, y=64)'.
4. Do not mention velocity or object interactions in the creation reasons.
"""

interpret_velocity_3_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 2 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0)
Interaction -- player object (id = 0) is touching car object (id = 2),
car object (id = 2) with new x-axis velocity = -3 and new y-axis velocity = +2,

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to -3
- The player object (id = 0) sets y-axis velocity to +1

Example reasons:
1. The player objects that touch a car object set their x-axis velocity to the car object's new x-axis velocity
2. The player objects that touch a car object set their y-axis velocity to the car object's new y-axis velocity minus 1.

Please output a list of 2 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
"""

interpret_velocity_4_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +2
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects that touch a ladder object with zero x-axis velocity set their x-axis velocity to +2
2. The player objects that touch a ladder object with positive y-axis velocity set their y-axis velocity to -4
3. The player objects that touch an unknown object with zero x-axis velocity set their x-axis velocity to +2
4. The player objects that touch an unknown object with positive y-axis velocity set their y-axis velocity to -4

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. always mention one object that {obj_type} objects are touching like in the example. If there is no interactions, please explicitly say that the {obj_type} objects are not touching anything
4. Only refer to the input velocities only as positive, negative, or zero values. Do not use equal, less than, or more than.
"""

interpret_no_int_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects set their x-axis velocity to +0
2. The player objects set their y-axis velocity to -4
3. The player objects are not deleted.
4. No player objects are created.

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_2_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with x-axis velocity = +0 and y-axis velocity +2,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. The player objects with x-axis velocity = +0 that touch a ladder object their x-axis velocity to +0
2. The player objects with y-axis velocity = +2 that touch a ladder object set their y-axis velocity to -4
3. The player objects with x-axis velocity = +0 that touch an unknown object their x-axis velocity to +0
4. The player objects with y-axis velocity = +2 that touch an unknown object set their y-axis velocity to -4

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_3_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0),
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),
Head-up-display: key_hud, sword_hud

Example output list of object changes:
- The player object (id = 0) sets x-axis velocity to +0
- The player object (id = 0) sets y-axis velocity to -4

Example reasons:
1. If key_hud exists, the player objects that touch a unknown object set their x-axis velocity to +0
2. If sword_hud exists, the player objects that touch a unknown object set their y-axis velocity to -4
3. If sword_hud exists, the player objects that touch a unknown object set their x-axis velocity to +0
4. If key_hud exists, the player objects that touch a unknown object set their y-axis velocity to -4

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
"""

interpret_4_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with player objects:
Example input list of objects:
player object (id = 0) with danger attribute = 8,
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The player object (id = 0) is deleted
- A new player object is created at (x=76,y=73)

Example reasons:
1. The player objects with danger attribute more than 5 that touch a ladder object are deleted
2. The player objects with danger attribute more than 5 that touch an unknown object are deleted
3. If player objects with danger attribute more than 5 touch a ladder object, a new player object is created at (x=76,y=73)
4. If player objects with danger attribute more than 5 touch an unknown object, a new player object is created at (x=76,y=73)

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. pick a "danger attribute more than" that is slightly lower than the actual danger attribute of the player object like in the example.
"""

interpret_5_prompt = \
"""\
I'll give you an input list of objects and an output list of object changes, and I want you to list 4 possible reasons for the effects

Here's an example with sword_hud objects:
Example input list of objects:
sword_hud object (id = 5),
Interaction -- player object (id = 0) is touching ladder object (id = 2),
Interaction -- player object (id = 0) is touching unknown object (id = 4),

Example output list of object changes:
- The sword_hud object (id = 5) is deleted
- A new sword_hud object is created at (x=96, y=37)
- A new sword_hud object is created at (x=32, y=34)
- A new sword_hud object is created at (x=64, y=64)

Example reasons:
1. If a player object touches an unknown object, the sword_hud objects are deleted
2. If a player object touches a ladder object, the sword_hud objects are deleted
3. If a player object touches an unknown object, new sword_hud objects are created at (x=96, y=37), (x=32, y=34), and (x=64, y=64)
4. If a player object touches a ladder object, new sword_hud objects are created at (x=96, y=37), (x=32, y=34), and (x=64, y=64)

Please output a list of 4 reasons of the {obj_type} objects for the following input and output list of objects.

Input list of objects:
{input}

Output list of object changes:
{effects}

Please follow these rules for your output:
1. make sure each reason only talks about one object change
2. do not talk about IDs
3. If objects are created at several positions, ALWAYS put them together in a single sentence. For example, 'If a player object touches a ladder object, new sword_hud objects are created at (x=96, y=37), (x=32, y=34), and (x=64, y=64)'.
"""

translate_4_prompt = """\
Given these classes:

class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int): x-axis velocity of the object
        velocity_y (int): y-axis velocity of the object
        falling_time (int): falling time of the object

    Methods:
        touches(obj: Obj) -> bool:
            Returns whether this Obj is touching the input obj (True/False)

class ObjList:
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

We want to output a function returns True when
"{description}"

The function should be of the form:
def get_{obj_type}_objects(obj_list: ObjList) -> bool:
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    res = False
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        pass
    return res

Output a function of this form with different conditions with the following rules:
1. Use only greater-than-or-equal-to sign (>=) for the conditions on falling time
2. Put the function in a python block
"""

non_creation_starter_program = """\
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    obj_list = obj_list.deepcopy() # make a new copy of obj_list
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        {obj_type}_obj.deleted = RandomValues([0])
    return obj_list"""

creation_starter_program = """\
def alter_{obj_type}_objects(obj_list: ObjList, action: str) -> ObjList:
    obj_list = obj_list.deepcopy() # make a new copy of obj_list
    obj_list = obj_list.create_object(np.asarray([[0, 0]]), '{obj_type}', Position(x=RandomValues(obj_list.grid_size), y=RandomValues(obj_list.grid_size)))
    {obj_type}_objs = obj_list.get_objs_by_obj_type('{obj_type}') # get all Obj of color '{obj_type}'
    for {obj_type}_obj in {obj_type}_objs: # {obj_type}_obj is of type Obj
        {obj_type}_obj.deleted = RandomValues([1])
    return obj_list"""

danger_att_prompt = """\
I want to know what are some important factors that could to a player's death in a video game. 
Please list 4 non-negative integer attributes that can be used to determine the player's death based on 
the player's history of velocity_x, velocity_y.

For each attribute, try to pick one that relies on a single SPECIFIC value of velocity_x or velocity_y, not just a range of values.

Then output a numbered list of 4 python functions, each function in its OWN code block, with the name compute_danger_attribute(obj: Obj) -> int that computes these attributes based on the player's history over the LATEST 5 timesteps.

The support class (this class is given, do not include it in your answer) is
class Obj:
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        history (dict): history of the object, containing the keys 'velocity_x', 'velocity_y'.
        
An example history of when the player dies:
{lst_txt}
"""

# Put each of the 4 functions in its own code block, with all of them having the signature compute_attribute(obj: Obj) -> int
