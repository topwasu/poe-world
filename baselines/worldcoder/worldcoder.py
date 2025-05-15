import logging

from classes.helper import StateTransitionTriplet, are_two_obj_lists_equal, ObjList, ObjTypeInteractionSelector
from learners.models import Model, Constraints
from learners.world_model_learner import WorldModelLearner
from classes.envs.object_tracker import ObjectTracker

from .from_world_coder.llm_utils import LLM
from .from_world_coder.synthesize_transit import synthesize_transit
from .from_world_coder.evaluator import get_transit_func

log = logging.getLogger('main')


montezuma_world_model_s0 = """\
def transition(state, event):
    new_state = state
    player = new_state.get_objs_by_obj_type("player")[0]
    def reset_player_velocity():
        player.velocity_x = 0
        player.velocity_y = 0
    def update_intrinsic_movements():
        moving_objs = new_state.get_objs_by_obj_type("skull")
        for obj in moving_objs:
            obj.velocity_x = obj.velocity_x or 1
    def check_collision_with_type(obj_type, side, percent):
        return any(player.touches(obj, side, percent)
                   for obj in new_state.get_objs_by_obj_type(obj_type))
    def handle_up():
        on_ladder = check_collision_with_type("ladder", 2, 0.5)
        on_rope = check_collision_with_type("rope", 2, 0.3)
        player.velocity_y = (-3 * on_ladder) or (-2 * on_rope)
    def handle_down():
        player.velocity_y = 3
    def handle_horizontal_movement(direction):
        blocked_right = direction > 0 and check_collision_with_type("barrier", 1, 0.5)
        blocked_left = direction < 0 and check_collision_with_type("barrier", 0, 0.5)
        player.velocity_x = 3 * direction * (not (blocked_right or blocked_left))
    def handle_fire(direction):
        bullet_velocity = 5
        bullet_start_x = player.x + (direction * 10)
        bullet_start_y = player.y
        # Create a bullet and set its velocity
        bullet = new_state.create_object("bullet", bullet_start_x, bullet_start_y).objs[-1]
        bullet.velocity_x = bullet_velocity * direction
    action_handlers = {
        "UP": handle_up,
        "DOWN": handle_down,
        "RIGHT": lambda: handle_horizontal_movement(1),
        "LEFT": lambda: handle_horizontal_movement(-1),
        "RIGHTFIRE": lambda: (handle_horizontal_movement(1), handle_fire(1)),
        "LEFTFIRE": lambda: (handle_horizontal_movement(-1), handle_fire(-1)),
        "FIRE": lambda: handle_fire(1),
        "NOOP": lambda: None
    }
    reset_player_velocity()
    action_handlers.get(event, lambda: None)()
    update_intrinsic_movements()
    return new_state
"""

montezuma_world_model_s1 = """\
def transition(state, event):
    new_state = state
    player = state.get_objs_by_obj_type("player")[0]
    skulls = state.get_objs_by_obj_type("skull")
    platforms = state.get_objs_by_obj_type("platform")
    ladders = state.get_objs_by_obj_type("ladder")
    conveyer_belts = state.get_objs_by_obj_type("conveyer_belt")
    def handle_noop_action():
        for platform in platforms:
            player_on_platform = player.touches(platform, 3, 1.0)
            player.velocity_y = 0
        player.velocity_x = 0
        apply_conveyor_effect()
    def apply_conveyor_effect():
        for conveyer_belt in conveyer_belts:
            player_on_conveyer = player.touches(conveyer_belt, 3, 1.0)
            player.velocity_x += conveyer_belt.velocity_x
    def handle_movement(delta_x, delta_y):
        player.velocity_x = delta_x
        player.velocity_y = delta_y
    def handle_right_action():
        handle_movement(3, player.velocity_y)
        apply_conveyor_effect()
    def handle_left_action():
        handle_movement(-3, player.velocity_y)
        apply_conveyor_effect()
    def handle_up_action():
        on_ladder = any(player.touches(ladder, 3, 1.0) for ladder in ladders)
        for ladder in ladders:
            on_ladder = player.touches(ladder, 3, 1.0)
            player.velocity_y = -3
            player.velocity_x = 0
            break
        else:
            player.velocity_y = -8
            player.velocity_x = 1
    def handle_down_action():
        for ladder in ladders:
            on_ladder = player.touches(ladder, 2, 0.5) or player.touches(ladder, 0, 1.0)
            player.velocity_y = 3
    def handle_skull_movement():
        for skull in skulls:
            skull.velocity_x = skull.velocity_x
    def handle_rightfire_action():
        handle_right_action()
        # Implement firing mechanism; for example, create a projectile object with a specific velocity
        projectiles = state.get_objs_by_obj_type("projectile")
        for projectile in projectiles:
            projectile.velocity_x = 5  # Assuming the projectile moves horizontally to the right
            projectile.velocity_y = 0
    # Map actions to handlers
    event_handlers = {
        'NOOP': handle_noop_action,
        'LEFT': handle_left_action,
        'RIGHT': handle_right_action,
        'UP': handle_up_action,
        'DOWN': handle_down_action,
        'RIGHTFIRE': handle_rightfire_action
    }
    # Call the appropriate handler based on the event
    event_handlers.get(event, lambda: None)()
    # Handle objects like skulls independently of event
    handle_skull_movement()
    return new_state
"""

montezuma_world_model_s2 = """\
def handle_noop_action():
    # Typically means maintaining velocity for skulls and other object behaviors
    for skull in skulls:
        skull.velocity_x = skull.velocity_x  # Skull continues its velocity
def transition(state, event):
    # Clone state to new_state
    new_state = state
    player = new_state.get_objs_by_obj_type("player")[0]
    skulls = new_state.get_objs_by_obj_type("skull")
    conveyer_belts = new_state.get_objs_by_obj_type("conveyer_belt")
    walls = new_state.get_objs_by_obj_type("wall")
    ladders = new_state.get_objs_by_obj_type("ladder")
    ropes = new_state.get_objs_by_obj_type("rope")
    platforms = new_state.get_objs_by_obj_type("platform")
    # Function to check wall collision
    def is_touching_wall(obj, direction):
        return any(obj.touches(wall, direction, 0.6) for wall in walls)
    # Update skull behavior
    def update_skull_behavior():
        for skull in skulls:
            touching_left_wall = is_touching_wall(skull, 0)
            touching_right_wall = is_touching_wall(skull, 1)
            # Reverse direction upon collision with walls
            skull.velocity_x = -skull.velocity_x * int(touching_left_wall or touching_right_wall) or skull.velocity_x
            # If not moving, set a default movement
            skull.velocity_x = skull.velocity_x or (1 * int(not touching_left_wall and not touching_right_wall))
    # Handle actions
    def handle_noop_action():
        # Ensure skulls are continually updated
        update_skull_behavior()
        # Check for interaction with conveyer belts
        for belt in conveyer_belts:
            on_belt = player.touches(belt, 3, 0.6)
            player.velocity_x += belt.velocity_x * on_belt
    def handle_left_action():
        player.velocity_x = -3 * (not is_touching_wall(player, 0))
        player.velocity_y = 0
        update_skull_behavior()
    def handle_right_action():
        player.velocity_x = 3 * (not is_touching_wall(player, 1))
        player.velocity_y = 0
        update_skull_behavior()
    def handle_up_action():
        on_ladder_or_rope = any(player.touches(ladder, 2, 0.6) for ladder in ladders) or any(player.touches(rope, 2, 0.6) for rope in ropes)
        player.velocity_y = -3 * on_ladder_or_rope
        player.velocity_x = 0
        update_skull_behavior()
    def handle_down_action():
        on_platform = any(player.touches(platform, 3, 0.6) for platform in platforms)
        player.velocity_y = 3 * (not on_platform)
        player.velocity_x = 0
        update_skull_behavior()
    def handle_fire_action():
        player.velocity_y -= 1
        update_skull_behavior()
    # Define action mapping
    action_map = {
        'LEFT': handle_left_action,
        'RIGHT': handle_right_action,
        'UP': handle_up_action,
        'DOWN': handle_down_action,
        'NOOP': handle_noop_action,
        'RIGHTFIRE': lambda: (handle_right_action(), handle_fire_action()),
        'LEFTFIRE': lambda: (handle_left_action(), handle_fire_action()),
        'FIRE': handle_fire_action
    }
    # Execute the appropriate action
    action_func = action_map.get(event, handle_noop_action)
    action_func()
    return new_state
"""

pong_world_model_s0 = """\
def transition(state, event):
    new_state = state
    def process_player(player, event):
        # Update player's velocity based on the event
        max_velocity_y = 9
        min_velocity_y = -9
        up_pressed = (event == 'UP')
        down_pressed = (event == 'DOWN')
        rightfire_pressed = (event == 'RIGHTFIRE')
        player.velocity_y += 1 * up_pressed
        player.velocity_y -= 1 * down_pressed
        player.velocity_y -= 3 * rightfire_pressed
        # Ensure velocities are within bounds
        player.velocity_y = max(min(player.velocity_y, max_velocity_y), min_velocity_y)
        # Deceleration for NOOP
        noop = (event == 'NOOP')
        player.velocity_y -= 1 * (player.velocity_y > 0 and noop)
        player.velocity_y += 1 * (player.velocity_y < 0 and noop)
    def process_ball_and_wall(ball):
        # Detect and handle wall collisions
        walls = new_state.get_objs_by_obj_type("wall")
        for wall in walls:
            for side in [0, 1]:  # Check for left and right collision
                touch = ball.touches(wall, side, 1.0)
                ball.velocity_x = -ball.velocity_x * touch + ball.velocity_x * (1 - touch)
            for side in [2, 3]:  # Check for up and down collision
                touch = ball.touches(wall, side, 1.0)
                ball.velocity_y = -ball.velocity_y * touch + ball.velocity_y * (1 - touch)
        # Initiate motion logic for static ball
        baseline_velocity_update = 3
        ball.velocity_x += (ball.velocity_x == 0) * baseline_velocity_update
        ball.velocity_y += (ball.velocity_y == 0) * baseline_velocity_update
    def process_enemy(enemy):
        # Restrict enemy velocities
        max_velocity_y = 6
        min_velocity_y = -6
        # Ensure enemy velocities are within bounds
        enemy.velocity_y = max(min(enemy.velocity_y, max_velocity_y), min_velocity_y)
        # Apply some deceleration logic
        deceleration_factor = 2
        enemy.velocity_y -= deceleration_factor * (enemy.velocity_y > 0)
        enemy.velocity_y += deceleration_factor * (enemy.velocity_y < 0)
        # Add a baseline update to enemy velocity to avoid stagnation
        baseline_enemy_velocity_update = 4
        enemy.velocity_y += (enemy.velocity_y == 0) * baseline_enemy_velocity_update
    # Apply processor based on object and event
    balls = new_state.get_objs_by_obj_type("ball")
    for ball in balls:
        process_ball_and_wall(ball)
    enemies = new_state.get_objs_by_obj_type("enemy")
    for enemy in enemies:
        process_enemy(enemy)
    players = new_state.get_objs_by_obj_type("player")
    for player in players:
        process_player(player, event)
    return new_state

"""

pong_world_model_s1 = """\
def transition(state, action):
    new_state = state
    player = new_state.get_objs_by_obj_type("player")[0]
    balls = new_state.get_objs_by_obj_type("ball")
    enemies = new_state.get_objs_by_obj_type("enemy")
    walls = new_state.get_objs_by_obj_type("wall")
    zones = new_state.get_objs_by_obj_type("zone")
    def apply_gravity_to_player():
        # Apply gravity to player (subtracting from velocity_y simulates gravity)
        player.velocity_y -= 2
    def apply_gravity_to_enemies():
        for enemy in enemies:
            # Apply gravity (-2 to velocity_y) unless enemy is already at max fall speed
            enemy.velocity_y = max(-6, enemy.velocity_y - 2)
    def handle_wall_collisions():
        for wall in walls:
            # Check player-wall collision
            player_touching_wall = player.touches(wall, 2, 1.0) or player.touches(wall, 3, 1.0)
            # Adjust player velocity when touching the wall
            # Set player velocity_y to 0 only when touching downward on a wall 
            # (simulating inability to fall through or move hence setting y-component of velocity to 0)
            player.velocity_y = 0
    def handle_zone_effects():
        for zone in zones:
            for ball in balls:
                # If the ball touches the zone, reverse its x velocity
                while ball.touches(zone, 0, 0.2) or ball.touches(zone, 1, 0.2):
                    ball.velocity_x = -ball.velocity_x
                    break
    def handle_noop_action():
        apply_gravity_to_enemies()
        apply_gravity_to_player()
        handle_zone_effects()
        handle_wall_collisions()
    def handle_right_action():
        player.velocity_y -= 3
        apply_gravity_to_enemies()
    def handle_left_action():
        player.velocity_y += 3
        apply_gravity_to_enemies()
    def handle_leftfire_action():
        player.velocity_y += 14
    def handle_rightfire_action():
        player.velocity_y -= 5
        apply_gravity_to_enemies()
    action_map = {
        "NOOP": handle_noop_action,
        "RIGHT": handle_right_action,
        "LEFT": handle_left_action,
        "LEFTFIRE": handle_leftfire_action,
        "RIGHTFIRE": handle_rightfire_action,
    }
    action_map.get(action, handle_noop_action)()
    return new_state
"""

pong_world_model_s2 = """\
def transition(state, event):
    new_state = state
    # Retrieve objects from the state
    player = state.get_objs_by_obj_type("player")[0]  
    enemies = state.get_objs_by_obj_type("enemy")
    balls = state.get_objs_by_obj_type("ball")
    def adjust_enemy_velocity():
        for enemy in enemies:
            # Increase or decrease velocity_y according to its sign or reverse it
            enemy.velocity_y += 2 * (1 - 2 * (enemy.velocity_y < 0))
            enemy.velocity_y = max(min(enemy.velocity_y, 6), -6)
    def continue_ball_movement():
        # Ball velocities to continue their direction and magnitude
        for ball in balls:
            ball.velocity_x, ball.velocity_y = ball.velocity_x, ball.velocity_y
    def adjust_velocity_noop():
        # For NOOP, apply logic where player decreases its velocity towards zero similarly
        player.velocity_y -= max(min(player.velocity_y, 2), -2)
        adjust_enemy_velocity()
        continue_ball_movement()
    def fire_event():
        player.velocity_y = 0
        for ball in balls:
            continue_ball_movement()
    # Dictionary for handling different actions
    events_dict = {
        "NOOP": adjust_velocity_noop,
        "FIRE": fire_event,
    }
    event_handler = events_dict.get(event, lambda: None)
    event_handler()
    return new_state
"""

popp_constraints = [
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


class WorldCoderModel(Model):
    def __init__(self, config, code):
        self.config = config
        self.code = code
        self.compilation_error, self.func_name, self.transit_func, self.exec_globals = get_transit_func(code)
        self.memory = None
        self.cache = {}
        self.cache_enabled = False
        
        if self.config.task.startswith('MontezumaRevenge'):
            self.constraints = Constraints('player', ObjTypeInteractionSelector('player'))
            self.constraints.rules = popp_constraints
            self.constraints.prepare_callables()
        else:
            self.constraints = Constraints('player', ObjTypeInteractionSelector('player'))

    def sample_next_scene(self,
                          obj_list_prev: ObjList,
                          event: str,
                          **kwargs) -> ObjList:
        obj_list_next = obj_list_prev.deepcopy()
        
        # step 1: pre-step
        obj_list_next.pre_step()

        # step 2: call the llm-generated code
        try:
            obj_list_next = self.transit_func(obj_list_next, event)
        except Exception as e:
            pass
        #TODO: include exec_globals later

        # step 3: post-step (update prev_x and prev_y)
        obj_list_next.step()
        
        # step 4: track the object
        object_tracker = ObjectTracker(init_obj_list=obj_list_prev)
        object_tracker.update(obj_list_next)

        # return the next scene
        return obj_list_next
    
    def clear_cache(self) -> None:
        self.cache = {}

    def enable_cache(self) -> None:
        self.clear_cache()
        self.cache_enabled = True

    def disable_cache(self) -> None:
        self.clear_cache()
        self.cache_enabled = False
        
    def clear_precompute_dist(self) -> None:
        pass
    
    def prepare_callables(self) -> None:
        self.constraints.prepare_callables()
    
    def remove_callables(self) -> 'WorldCoderModel':
        new_model = WorldCoderModel(self.config, self.code)
        new_model.constraints.callables = []
        return new_model
    
    def get_features(self, obj_list):
        return self.constraints.get_features(obj_list)


class WorldCoder(WorldModelLearner):
    def __init__(self, config):
        self.config = config
        self.world_model_code = None
        self.saved_world_model_code = None
        self.world_model = None
        self.c = []
        
        self.update_world_model_budget = 10.0

    def synthesize_world_model(self, c: list[StateTransitionTriplet], **kwargs) -> WorldCoderModel:
        """
        :param c: List of StateTransitionTriplet
        :return: WorldCoderModel
        """
        self.c = c
        
        if self.config.task.startswith('MontezumaRevenge') and self.config.seed == 0:
            self.world_model_code = montezuma_world_model_s0
        elif self.config.task.startswith('MontezumaRevenge') and self.config.seed == 1:
            self.world_model_code = montezuma_world_model_s1
        elif self.config.task.startswith('MontezumaRevenge') and self.config.seed == 2:
            self.world_model_code = montezuma_world_model_s2
        elif self.config.task.startswith('Pong') and self.config.seed == 0:
            self.world_model_code = pong_world_model_s0
        elif self.config.task.startswith('Pong') and self.config.seed == 1:
            self.world_model_code = pong_world_model_s1
        elif self.config.task.startswith('Pong') and self.config.seed == 2:
            self.world_model_code = pong_world_model_s2
        else:
            llm = LLM(seed=self.config.seed)
            res = synthesize_transit(self.c, llm=llm, max_budget=10.0, np_rng=self.config.seed)
            self.world_model_code = res['code']
            
            log.info(f"Synthesized world model code: {self.world_model_code}")
            
        self.world_model = WorldCoderModel(self.config, self.world_model_code)
        return self.world_model

    def update_world_model(self, c, fast=True, **kwargs):
        self.c = self.c + c
        
        if self.update_world_model_budget <= 0: 
            return self.world_model
        
        llm = LLM(seed=self.config.seed)
        
        spending = 0.1 if fast else 0.5
        res = synthesize_transit(self.c, init_transit_code=self.world_model_code, llm=llm, max_budget=min(spending, self.update_world_model_budget), with_total_cost=True, np_rng=self.config.seed)
        self.world_model_code = res['code']
        
        self.update_world_model_budget -= res['final_total_cost']
        
        self.world_model = WorldCoderModel(self.config, self.world_model_code)
        return self.world_model
    
    def save_snapshot(self):
        self.saved_world_model_code = self.world_model_code
        
    def load_snapshot(self):
        self.world_model_code = self.saved_world_model_code
        self.world_model = WorldCoderModel(self.config, self.world_model_code)
        return self.world_model
