import math
import logging

from classes.helper import ObjListWithMemory, Constants
from learners.models import WorldModel

log = logging.getLogger('main')


def manual_heuristics_factory(target_abstract_state, config):
    targets = [(idx, int(x))
               for idx, x in enumerate(target_abstract_state[1:-1].split(', '))
               if int(x) != -1]

    def f(game_state, n):
        sm = 0
        cur_obj_list = game_state.cur_obj_list
        cur_abstract_state = game_state.get_abstract_state()
        if cur_abstract_state == target_abstract_state:
            return 1000
        satisfieds = [
            (idx, int(x))
            for idx, x in enumerate(cur_abstract_state[1:-1].split(', '))
            if int(x) != -1
        ]
        for target in targets:
            # Already satisfied target, then skip
            if target in satisfieds:
                continue

            try:
                obj = [x for x in cur_obj_list.objs if x.id == target[1]][0]
            except:
                return -1000
            
            try:
                player_obj = cur_obj_list.get_objs_by_obj_type('player')[0]
            except:
                return -1000

            # Compute heuristic value
            if config.heuristics == 'advanced':
                sm += max(0, obj.left_side - player_obj.center_x) + max(
                    0, player_obj.center_x - obj.right_side) + max(
                        0, obj.top_side - player_obj.center_y) + max(
                            0, player_obj.center_y - obj.bottom_side)
            elif config.heuristics == 'basic':
                sm += abs(obj.center_x -
                          player_obj.center_x) + abs(obj.center_y -
                                                     player_obj.center_y)
            else:
                raise NotImplementedError
        # return 1 / (sm + n + 1)
        return -sm

    return f


class Node:
    def __init__(self, state: 'GameState', parent=None, last_action_seq=None):
        self.state = state  # Game state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times node was visited
        self.value = -1000  # Value of the node
        self.last_action_seq = last_action_seq

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.old_get_legal_action_seqs())

    def best_child(self, exploration_weight=1.0):
        """Get the best child node based on UCT."""
        if not self.children:
            raise Exception("No children nodes to select from!")
        weights = [
            # (child.value / (child.visits + 1e-6)) +
            (child.value) + exploration_weight *
            math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        # but there might be no weights if not choose terminal children
        return self.children[max(index for index, v in enumerate(weights)
                                 if v == max(weights))]

    def expand(self):
        """Expand a node by adding a child for an untried action."""
        tried_action_seqs = [child.last_action_seq for child in self.children]
        legal_action_seqs = self.state.old_get_legal_action_seqs()
        for action_seq in legal_action_seqs:
            if action_seq not in tried_action_seqs:
                next_state = self.state.old_perform_action_seq(action_seq)
                child_node = Node(next_state, self, action_seq)
                self.children.append(child_node)
                return child_node
        raise Exception("No actions to expand")

    def backpropagate(self, value):
        """Update the node and its ancestors with the simulation result."""
        self.visits += 1
        # self.value += value
        self.value = max(self.value, value)
        if self.parent:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, config):
        self.config = config
        self.exploration_weight = self.config.mcts.exploration_weight
        
    def search(self, *args, **kwargs):
        res = None
        if kwargs['iterations'] != 0:
            res = self.search_old(*args, **kwargs)
        if res is None or (isinstance(res, tuple) and res[0] is None):
            return self._search(*args, **kwargs, future_length=self.config.mcts.future_length)
        return res
        
    def _search(self,
               cur_obj_list: ObjListWithMemory,
               target_abstract_state: str,
               world_model: WorldModel,
               iterations: int = 2000,
               target_id: int = None,
               ret_concrete_state: bool = False,
               future_length: int = 1):
        world_model.enable_cache()
        initial_state = GameState(world_model,
                                  cur_obj_list,
                                  target_abstract_state,
                                  self.config,
                                  target_id=target_id)
        heuristics_f = manual_heuristics_factory(target_abstract_state,
                                                 self.config)
        root = Node(initial_state)

        goal = None
        
        # Greedy search to get to the goal
        node = root
        actions = []
        ct = 0
        best_action_seq = None
        while not node.state.is_goal() and ct < self.config.mcts.n_tries:
            if node.state.is_terminal():
                break
                
            legal_action_seqs = node.state.new_get_legal_action_seqs(length=future_length)
            best_action_seq = None
            best_heuristic = float('-inf')
            for action_seq in legal_action_seqs:
                next_state = node.state.new_perform_action_seq(action_seq)
                heuristic_value = heuristics_f(next_state, len(actions + action_seq))
                if heuristic_value > best_heuristic and not next_state.died:
                    best_heuristic = heuristic_value
                    best_action_seq = action_seq

            if best_action_seq is None:
                # log.info(f'No valid action sequence found')
                # log.info(f'Backtracking')
                if node.last_action_seq is None:
                    break
                best_action_seq = node.last_action_seq[:1]
                actions = actions[:-future_length]
                node = node.parent
                
            # best_action_seq = best_action_seq[:1]
            node = Node(node.state.new_perform_action_seq(best_action_seq), node, best_action_seq)
            actions = actions + best_action_seq
            node.parent.children.append(node)
            
            # log.info(f'Search iteration {ct}: {actions} {best_heuristic} {node.state.get_abstract_state()}')
            # log.info(f'Player position: {node.state.cur_obj_list.get_objs_by_obj_type("player")[0].center_x} {node.state.cur_obj_list.get_objs_by_obj_type("player")[0].center_y}')
            # log.info(f'Skull position: {node.state.cur_obj_list.get_objs_by_obj_type("skull")[0].center_x} {node.state.cur_obj_list.get_objs_by_obj_type("skull")[0].center_y}')
            ct += 1
        
        if node.state.is_goal():
            goal = node
        
        if goal is None:
            # log.info(f'No goal found in {ct} iterations')
            if ret_concrete_state:
                return None, None
            return None

        current_node = goal
        reversed_plan = []
        while current_node.parent is not None:
            reversed_plan.append(current_node.last_action_seq)
            current_node = current_node.parent

        if ret_concrete_state:
            return sum(reversed_plan[::-1], []), goal.state.cur_obj_list
        return sum(reversed_plan[::-1], [])

    def search_old(self,
               cur_obj_list: ObjListWithMemory,
               target_abstract_state: str,
               world_model: WorldModel,
               iterations: int = 2000,
               target_id: int = None,
               ret_concrete_state: bool = False):
        world_model.enable_cache()
        initial_state = GameState(world_model,
                                  cur_obj_list,
                                  target_abstract_state,
                                  self.config,
                                  target_id=target_id)
        heuristics_f = manual_heuristics_factory(target_abstract_state,
                                                 self.config)
        root = Node(initial_state)

        goal = None

        for ct in range(iterations):
            w = 10**(ct // 1000)

            node = root

            actions = []
            # Selection: Navigate tree until leaf node
            while (not node.state.is_terminal()) and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight * w)
                actions = actions + node.last_action_seq

            # Expansion: Expand the node
            if not node.state.is_terminal():
                node = node.expand()
                actions = actions + node.last_action_seq

            # Simulation: Perform a rollout to get a result
            result = heuristics_f(node.state, len(actions))

            log.debug(
                f'MCTS iteration {ct}: {actions} {result} {node.state.get_abstract_state()}'
            )

            if node.state.is_goal():
                goal = node
                break

            # Backpropagation: Propagate the result back up the tree
            node.backpropagate(result)

        log.info(f'Did MCTS for {ct} iterations')

        world_model.disable_cache()

        if goal is None:
            if ret_concrete_state:
                return None, None
            return None

        current_node = goal
        reversed_plan = []
        while current_node.parent is not None:
            reversed_plan.append(current_node.last_action_seq)
            current_node = current_node.parent

        if ret_concrete_state:
            return sum(reversed_plan[::-1], []), goal.state.cur_obj_list
        return sum(reversed_plan[::-1], [])


# Define the game-specific state class
class GameState:
    def __init__(self,
                 world_model: WorldModel,
                 cur_obj_list: ObjListWithMemory,
                 target_abstract_state,
                 config,
                 died=False,
                 depth=0,
                 target_id=None):
        self.world_model = world_model
        self.target_abstract_state = target_abstract_state
        self.cur_obj_list = cur_obj_list
        self.config = config
        self.died = died
        self.depth = depth
        self.target_id = target_id

    def get_abstract_state(self):
        if self.target_id is not None:
            try:
                player_obj = self.cur_obj_list.get_objs_by_obj_type('player')[0]
                target_obj = self.cur_obj_list.get_obj_by_id(self.target_id)
            except:
                return '[-1]'
            return str([self.target_id
                        ]) if player_obj.overlaps(target_obj) else '[-1]'
        else:
            return str(self.world_model.get_features(self.cur_obj_list))

    def abstract_state(self, obj_list: ObjListWithMemory):
        if self.target_id is not None:
            try:
                player_obj = obj_list.get_objs_by_obj_type('player')[0]
                target_obj = obj_list.get_obj_by_id(self.target_id)
            except:
                return '[-1]'
            return str([self.target_id
                        ]) if player_obj.overlaps(target_obj) else '[-1]'
        else:
            return str(self.world_model.get_features(obj_list))

    def is_stable_state(self):
        if self.target_id is not None:
            return True
        new_obj_list = self.cur_obj_list.deepcopy()
        memory = new_obj_list.memory

        new_obj_list = self.world_model.sample_next_scene(new_obj_list,
                                                          'NOOP',
                                                          memory=memory,
                                                          det=self.config.det_world_model)
        
        return self.abstract_state(ObjListWithMemory(new_obj_list, memory)) == self.abstract_state(
            self.cur_obj_list)

    def old_get_legal_action_seqs(self):
        primitives = Constants.ACTIONS
        if self.config.mcts.sticky_actions:
            return [[primitive] * ct for primitive in primitives
                    for ct in [8, 4, 1]]
        else:
            return [[primitive] * ct for primitive in primitives
                    for ct in [1]]
            
    def old_perform_action_seq(self, action_seq):
        """Return the state resulting from performing an action."""
        cur_obj_list = self.cur_obj_list.deepcopy()  # Can we remove this?
        died = False
        memory = cur_obj_list.memory
        for action in action_seq:
            old_obj_list = cur_obj_list
            cur_obj_list = self.world_model.sample_next_scene(cur_obj_list,
                                                              action,
                                                              memory=memory,
                                                              det=self.config.det_world_model)
            memory.add_obj_list_and_action(old_obj_list, action)

            # Check if died
            player_objs = cur_obj_list.get_objs_by_obj_type('player')
            if len(player_objs
                   ) != len(old_obj_list.get_objs_by_obj_type('player')) or player_objs[0].history['deleted'][-2] == 1:
                died = True
                break
        return GameState(self.world_model,
                         ObjListWithMemory(cur_obj_list, memory),
                         self.target_abstract_state,
                         self.config,
                         died,
                         self.depth + 1,
                         target_id=self.target_id)
        
    def new_get_legal_action_seqs(self, length=1):
        primitives = Constants.ACTIONS
        return [[primitive] * ct for primitive in primitives
                for ct in [length]]

    def new_perform_action_seq(self, action_seq):
        """Return the state resulting from performing an action."""
        cur_obj_list = self.cur_obj_list.deepcopy()  # Can we remove this?
        memory = cur_obj_list.memory
        for action in action_seq:
            old_obj_list = cur_obj_list
            cur_obj_list = self.world_model.sample_next_scene(cur_obj_list,
                                                              action,
                                                              memory=memory,
                                                              det=self.config.det_world_model)
            memory.add_obj_list_and_action(old_obj_list, action)
            
            state = GameState(self.world_model,
                         ObjListWithMemory(cur_obj_list, memory),
                         self.target_abstract_state,
                         self.config,
                         False,
                         self.depth + 1,
                         target_id=self.target_id)

            # Check if died
            player_objs = cur_obj_list.get_objs_by_obj_type('player')
            if len(player_objs
                   ) != len(old_obj_list.get_objs_by_obj_type('player')) or player_objs[0].history['deleted'][-2] == 1:
                state.died = True
                break
            
            if state.is_goal():
                break
        return state

    def is_terminal(self):
        """Check if the game is over."""
        return self.died or self.is_goal()

    def is_goal(self):
        return (self.get_abstract_state() == self.target_abstract_state
                ) and self.is_stable_state() and (not self.died)

    def get_result(self):
        """Get the result of the game (a numeric score)."""
        raise NotImplementedError
