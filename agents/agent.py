from typing import List, Tuple, Dict, Any, Optional, Union
import re
import os
import time
import pickle
import uuid
import logging
import numpy as np
import hashlib
from collections import defaultdict

from agents.mcts import MCTS
from agents.utils import *
from classes.envs import *
from classes.helper import StateTransitionTriplet, ObjListWithMemory, Constants, GameState
from classes.envs.renderer import get_image_renderer, get_human_renderer
from learners.world_model_learner import WorldModelLearner

log = logging.getLogger('main')


class Agent:
    """
    Main agent class that handles high-level planning and execution in the environment.
    Combines world model learning, MCTS planning, and abstract graph construction.
    """
    def __init__(self, config: Dict[str, Any],
                 world_learner: WorldModelLearner) -> None:
        """Initialize agent with configuration and world model learner"""
        self.config = config
        
        renderer = get_human_renderer(config)
        self.atari_env = create_atari_env(config, config.task, renderer=renderer)

        self.renderer = get_image_renderer(config)

        self.world_learner = world_learner

        self.mcts = MCTS(config)
        
        self.abstract_planning = True if len(self.world_learner.world_model.constraints) > 0 else False
        
        self.actions_taken = []
        self.n_build_graph_calls = 0
        self.slurm_job_ids = []  # Track SLURM job IDs
    def plan_and_execute(self, goal_obj_type: str = 'key') -> None:
        """
        Main execution loop that runs the full planning and execution pipeline.
        Attempts to reach an object with specified goal_id in the environment.
        
        Args:
            goal_id: ID of the target object to reach (default=1 for key)
        """
        log.info(f'Running pipeline to get object {goal_obj_type}')

        # Initialize environment
        cur_obj_list, cur_game_state = self._initialize_environment()
        n_budget_iterations = self.config.agent.initial_budget_iterations
        
        # Build initial abstract graph
        if self.abstract_planning:
            skills_hsh, achievables_hsh = self.build_graph(cur_obj_list.deepcopy(),
                                                        n_budget_iterations,
                                                        goal_obj_type,
                                                        load=True)

            n_goals_achieved = 0

            while True:
                
                if self.atari_env.n_reset > 10:
                    log.info(f'NO GOAL AFTER 10 EPISODES')
                    break
                    
                # Try to find and execute a plan
                # Note: skills_hsh and achievables_hsh may be updated here; they are mutable
                symbolic_plan = self._get_symbolic_plan(cur_obj_list, goal_obj_type,
                                                        skills_hsh,
                                                        achievables_hsh)
                
                log.info(f'All actions taken: ' + str(self.actions_taken))

                if symbolic_plan is None:
                    # Handle case where no plan found
                    (cur_obj_list, cur_game_state, skills_hsh, achievables_hsh,
                    n_budget_iterations) = self._handle_no_plan(
                        cur_obj_list, cur_game_state, goal_obj_type, skills_hsh,
                        achievables_hsh, n_budget_iterations)
                    
                    continue

                log.info(
                    f'Currently following this symbolic plan:\n{symbolic_plan}')

                # Execute the plan
                success, cur_obj_list, cur_game_state = self._execute_plan(
                    symbolic_plan, cur_obj_list,
                    cur_game_state, n_budget_iterations,
                    goal_obj_type, skills_hsh, achievables_hsh
                )

                if success:
                    log.info(f"GOT GOAL! ({n_goals_achieved+1}/{self.config.agent.n_goals_to_achieve})")
                    
                    log.info(f'All actions taken: ' + str(self.actions_taken))
                    
                    n_goals_achieved += 1
                    
                    if n_goals_achieved >= self.config.agent.n_goals_to_achieve:
                        break
        else:
            n_goals_achieved = 0
            while True:
                success = False
                
                goal_ids = self._get_goal_ids(cur_obj_list, goal_obj_type)
                
                ct = 0
                while len(goal_ids) == 0:
                    action = np.random.choice(Constants.ACTIONS)
                    cur_obj_list, cur_game_state = self.atari_env.step(action)
                    self.actions_taken.append(action)
                    ct += 1
                    goal_ids = self._get_goal_ids(cur_obj_list, goal_obj_type)
                log.info(f"Acting randomly for {ct} frames to see a goal object")
                
                success, cur_obj_list, cur_game_state = self._try_reach_multiple_goals(
                            goal_ids, cur_obj_list,
                            cur_game_state, n_budget_iterations)

                if success:
                    log.info(f"GOT GOAL! ({n_goals_achieved+1}/{self.config.agent.n_goals_to_achieve})")
                    
                    log.info(f'All actions taken: ' + str(self.actions_taken))
                    
                    n_goals_achieved += 1
                    
                    for _ in range(1):
                        action = np.random.choice(Constants.ACTIONS)
                        cur_obj_list, cur_game_state = self.atari_env.step(action)
                        self.actions_taken.append(action)
                        
                    log.info(f"Acting randomly for 1 frames after getting the goal")
                    
                    if n_goals_achieved >= self.config.agent.n_goals_to_achieve:
                        break
                else:
                    log.info(f'Failed to get goal')
                    log.info(f'All actions taken: ' + str(self.actions_taken))
                    
                    for _ in range(10):
                        action = np.random.choice(Constants.ACTIONS)
                        cur_obj_list, cur_game_state = self.atari_env.step(action)
                        self.actions_taken.append(action)
                        
                    log.info(f"Acting randomly for 10 frames to find a plan to get the goal")
                    
                if cur_game_state == GameState.GAMEOVER:
                    log.info("GAME OVER!")
                    log.info(f'All actions taken: ' + str(self.actions_taken))
                    break

    def _initialize_environment(self) -> Tuple[ObjListWithMemory, Any]:
        """Initialize the environment and handle ghost if needed"""
        cur_obj_list, cur_game_state = self.atari_env.reset()

        if self.config.agent.get_rid_of_ghost:
            ghost_removal_actions = self._get_ghost_removal_actions()
            for action in ghost_removal_actions:
                cur_obj_list, cur_game_state = self.atari_env.step(action)
                self.actions_taken.append(action)

        return cur_obj_list, cur_game_state

    def _get_ghost_removal_actions(self) -> List[str]:
        """Return sequence of actions to remove ghost"""
        return ['DOWN'] * 11 + ['RIGHTFIRE', 'LEFT', 'LEFT', 'RIGHT', 'DOWN',
                'LEFT', 'FIRE', 'UP'] + ['NOOP'] * 14 + \
               ['RIGHT'] * 16 + ['RIGHTFIRE'] * 6 + \
               ['NOOP', 'RIGHTFIRE', 'RIGHTFIRE'] + ['UP'] * 5 + \
               ['LEFTFIRE'] * 4 + ['DOWN'] * 2 + ['RIGHT', 'UP', 'RIGHT'] + \
               ['DOWN'] * 11 + ['LEFT'] * 15 + ['NOOP']
               
    def _get_goal_id(self, obj_list, goal_obj_type) -> int:
        """Get the ID of the goal object type"""
        goal_obj = obj_list.get_objs_by_obj_type(goal_obj_type)
        if goal_obj:
            return goal_obj[0].id
        else:
            return -1
        
    def _get_goal_ids(self, obj_list, goal_obj_type) -> List[int]:
        """Get the IDs of all goal objects of a given type"""
        goal_objs = obj_list.get_objs_by_obj_type(goal_obj_type)
        if goal_objs:
            return [obj.id for obj in goal_objs]
        else:
            return []

    def _get_symbolic_plan(
            self, cur_obj_list: Any, goal_obj_type: str, skills_hsh: Dict[Tuple[str,
                                                                          str],
                                                                    Any],
            achievables_hsh: Dict[Tuple[str, int],
                                  Any]) -> Optional[List[str]]:
        """Get symbolic plan from current state to goal"""
        goal_id = self._get_goal_id(cur_obj_list, goal_obj_type)
        if goal_id == -1:
            return None
        return self.search_and_prune_in_graph(
            self._abstract_state(cur_obj_list), goal_id, skills_hsh,
            achievables_hsh)[0]

    def _handle_no_plan(
        self, cur_obj_list: Any, cur_game_state: Any, goal_obj_type: str,
        skills_hsh: Dict[Tuple[str, str], Any],
        achievables_hsh: Dict[Tuple[str, int], Any], n_budget_iterations: int
    ) -> Tuple[Any, Any, Dict[Tuple[str, str], Any], Dict[Tuple[str, int],
                                                          Any], int]:
        """Handle case where no plan is found by waiting or increasing budget"""
        
        log.info("Can't find symbolic plan -- wait for 20 frames")
        # Try waiting to die
        for _ in range(20):
            cur_obj_list, cur_game_state = self.atari_env.step("NOOP")
            self.actions_taken.append("NOOP")
        log.info(f"Waited for 20 frames")
        log.info("Wait until there is goal object")
        # Then wait until there is goal object
        ct = 0
        while not cur_obj_list.get_objs_by_obj_type(goal_obj_type):
            cur_obj_list, cur_game_state = self.atari_env.step("NOOP")
            self.actions_taken.append("NOOP")
            ct += 1
        log.info(f"Waited for {ct} frames")
        
        if self._abstract_state(cur_obj_list) == self.null_abstract_state:
            log.info("No abstract state found -- resetting")
            cur_obj_list, cur_game_state = self.atari_env.reset()
            self.actions_taken.append("RESTART")

        symbolic_plan = self._get_symbolic_plan(cur_obj_list, goal_obj_type,
                                                skills_hsh, achievables_hsh)

        if symbolic_plan is None:
            log.info("Still can't find symbolic plan -- building new graph")

            ct = 0
            while symbolic_plan is None and self.atari_env.n_reset <= 10:
                # Increase budget and rebuild graph
                ct += 1
                if self.config.agent.budget_increase_mode == 'slow':
                    if ct % 5 == 0:
                        n_budget_iterations += \
                            self.config.agent.initial_budget_iterations
                elif self.config.agent.budget_increase_mode == 'fast':
                    n_budget_iterations += \
                            self.config.agent.initial_budget_iterations

                log.info(
                    f"Building new graph with budget {n_budget_iterations}"
                )
                skills_hsh, achievables_hsh = self.build_graph(
                    cur_obj_list.deepcopy(),
                    n_budget_iterations,
                    goal_obj_type,
                    load=False)
                symbolic_plan = self._get_symbolic_plan(
                    cur_obj_list, goal_obj_type, skills_hsh, achievables_hsh)
                
                if symbolic_plan is None:
                    log.info("Still can't find with the new graph -- resetting")
                    cur_obj_list, cur_game_state = self.atari_env.reset()
                    self.actions_taken.append("RESTART")

        return cur_obj_list, cur_game_state, skills_hsh, achievables_hsh, n_budget_iterations

    def _execute_plan(self, symbolic_plan: List[str], cur_obj_list: Any,
                      cur_game_state: Any, n_budget_iterations: int,
                      goal_obj_type: str, skills_hsh: Dict[Tuple[str, str], Any],
                      achievables_hsh: Dict[Tuple[str, int], Any]) -> bool:
        """Execute symbolic plan and try to reach goal"""
        prev_state = self._abstract_state(cur_obj_list)

        # Follow plan steps
        for target_state in symbolic_plan[1:]:
            log.info(
                f'Trying to get to this abstract state {target_state} from {prev_state}'
            )
            success, cur_obj_list, cur_game_state, = self._execute_plan_step(
                prev_state, target_state,
                cur_obj_list, cur_game_state,
                n_budget_iterations, skills_hsh
            )

            if not success:
                return False, cur_obj_list, cur_game_state

            prev_state = target_state

        # Try to reach final goal
        goal_id = self._get_goal_id(cur_obj_list, goal_obj_type)
        if goal_id == -1:
            log.info('No goal found')
            return False, cur_obj_list, cur_game_state
        return self._try_reach_goal(symbolic_plan[-1], goal_id, cur_obj_list,
                                    cur_game_state, n_budget_iterations,
                                    achievables_hsh)

    def _execute_plan_step(self, prev_state: str, target_state: str,
                           cur_obj_list: Any, cur_game_state: Any,
                           n_budget_iterations: int,
                           skills_hsh: Dict[Tuple[str, str], Any]) -> bool:
        """Execute single step of plan"""
        log.info(f'Trying to reach {target_state} from {prev_state}')

        cur_obj_list, cur_game_state, success, died_by_monster = \
            self.run_low_level(cur_obj_list, cur_game_state,
                             n_budget_iterations, target_state)

        if not success:
            log.info(
                f'Failed to reach {target_state} from {prev_state} (died_by_monster={died_by_monster})'
            )
            if not (self.config.agent.ignore_monster and died_by_monster):
                log.info(f'Removing edge {prev_state} --> {target_state}')
                del skills_hsh[(prev_state, target_state)]
        else:
            log.info(f'Succeed in getting to {target_state} from {prev_state}!')
        return success, cur_obj_list, cur_game_state 

    def _try_reach_goal(self, final_state: str, goal_id: int,
                        cur_obj_list: Any, cur_game_state: Any,
                        n_budget_iterations: int,
                        achievables_hsh: Optional[Dict[Tuple[str, int], Any]]) -> bool:
        """Try to reach final goal from last plan state"""
        log.info(f'Trying to get goal from {final_state}')

        if achievables_hsh is not None and (final_state, goal_id) not in achievables_hsh:
            return False

        cur_obj_list, cur_game_state, success, died_by_monster = \
            self.run_low_level(cur_obj_list, cur_game_state,
                             n_budget_iterations, str([goal_id]),
                             target_id=goal_id)

        if not success:
            log.info(f'Failed to get goal (died_by_monster={died_by_monster})')
            if achievables_hsh is not None and not (self.config.agent.ignore_monster and died_by_monster):
                log.info(f'Removing the edge from {final_state} to the goal')
                achievables_hsh.pop((final_state, goal_id))
        else:
            log.info('SUCCESS IN GETTING THE GOAL!')
        return success, cur_obj_list, cur_game_state
    
    def _try_reach_multiple_goals(self, goal_ids: List[int],
                                  cur_obj_list: Any, cur_game_state: Any,
                                  n_budget_iterations: int) -> bool:
        """Try to reach multiple goals from last plan state"""
        log.info(f'Trying to get these goals: {goal_ids}')
        
        if self.mcts.config.heuristics == 'both':
            raise NotImplementedError
        
        world_model = self.world_learner.world_model
        history_actions = []
        str_seed = str(goal_ids)
        rng = np.random.default_rng(
            (int(hashlib.md5(str_seed.encode()).hexdigest(), 16) + self.config.random_seed) % (2**32))
        success = False
        while not success:
            taken_actions = []
            # Search for a low-level plan
            # TODO: Can speed this up by not searching for all goals
            best_plan = None
            best_goal_id = None
            for goal_id in goal_ids:
                if goal_id not in [obj.id for obj in cur_obj_list]:
                    continue
                new_plan = self.mcts.search(cur_obj_list,
                                            str([goal_id]),
                                            world_model,
                                            iterations=n_budget_iterations,
                                            target_id=goal_id)
                if new_plan == []:
                    new_plan = None
                
                if new_plan is not None:
                    log.info(f'Found plan for goal {goal_id}')
                    if best_plan is None or len(new_plan) < len(best_plan):
                        best_plan = new_plan
                        best_goal_id = goal_id
            # If can't find plan, then stop trying
            log.info(f'Actions so far {history_actions}')
            if best_plan is None:
                log.info(f'Did not find plan')
                break
            else:
                log.info(f'Now following plan={best_plan}')
                log.info(f'to get to goal {best_goal_id}')

            for idx, action in enumerate(best_plan):
                cur_obj_list, cur_game_state = self.atari_env.step(action)
                self.actions_taken.append(action)
                taken_actions.append(action)
                history_actions.append(action)
                
                if self._abstract_state(
                    cur_obj_list, target_id=best_goal_id
                ) == str([best_goal_id]):
                    success = True
                    break

                # Check whether we are on a good path to reaching the target
                # Don't check too often though, it's slow
                if rng.random() < self.config.agent.short_term_accuracy:
                    if not self._on_good_path(cur_obj_list,
                                              best_plan[idx + 1:],
                                              str([best_goal_id]),
                                              world_model,
                                              target_id=best_goal_id):
                        log.info(f'Plan failed after taking {taken_actions}')
                        break
                    
            if success:
                log.info(f'Success in getting to {goal_ids}!')
                break
        
        return success, cur_obj_list, cur_game_state

    def run_low_level(
            self,
            cur_obj_list: Any,
            cur_game_state: Any,
            n_budget_iterations: int,
            target_abstract_state: str,
            target_id: Optional[int] = None) -> Tuple[Any, Any, bool, bool]:
        """
        Execute low-level actions to reach a target abstract state from current state.
        Uses MCTS to find action sequences and updates world model on failures.

        Args:
            cur_obj_list: Current list of objects in the environment
            cur_game_state: Current game state
            n_budget_iterations: Number of MCTS iterations to use
            target_abstract_state: Abstract state to reach
            target_id: Optional specific object ID to target
        
        Returns:
            Tuple of (final object list, final game state, success flag, died_by_monster flag)
        """

        # save a snapshot of world learner
        if self.config.agent.fast_world_update:
            self.world_learner.save_snapshot()

        history_actions = []
        world_model = self.world_learner.world_model
        success = False
        died_by_monster = False
        ct = 0

        cur_c = []

        str_seed = self._abstract_state(cur_obj_list) + target_abstract_state
        rng = np.random.default_rng(
            (int(hashlib.md5(str_seed.encode()).hexdigest(), 16) + self.config.random_seed) % (2**32))
        
        found_at_least_one_plan = False
        
        obj_lists, game_states = [cur_obj_list], [cur_game_state]
        taken_actions = []

        while ct < self.config.agent.max_iter:
            ct += 1
            
            start_index = len(obj_lists)

            died = False

            # This is to prevent sticky keys
            for _ in range(1):
                cur_obj_list, cur_game_state = self.atari_env.step('NOOP')
                self.actions_taken.append("NOOP")
                obj_lists.append(cur_obj_list.deepcopy())
                game_states.append(cur_game_state)
                taken_actions.append('NOOP')
                history_actions.append('NOOP')

                # If player dies, does not count
                if cur_game_state == GameState.RESTART:  # player died
                    died = True
                    break

            if died:
                log.info(f'Died trying to get to {target_abstract_state}!')
                break

            # Search for a low-level plan
            if self.mcts.config.heuristics == 'both':
                self.mcts.config.heuristics = 'basic'
                new_plan = self.mcts.search(cur_obj_list,
                                            target_abstract_state,
                                            world_model,
                                            iterations=n_budget_iterations,
                                            target_id=target_id)
                if new_plan is None:
                    self.mcts.config.heuristics = 'advanced'
                    new_plan = self.mcts.search(cur_obj_list,
                                                target_abstract_state,
                                                world_model,
                                                iterations=n_budget_iterations,
                                                target_id=target_id)
                self.mcts.config.heuristics = 'both'
            else:
                new_plan = self.mcts.search(cur_obj_list,
                                            target_abstract_state,
                                            world_model,
                                            iterations=n_budget_iterations,
                                            target_id=target_id)

            # If can't find plan, then stop trying
            log.info(f'Actions so far {history_actions}')
            if new_plan is not None:
                plan = new_plan
                found_at_least_one_plan = True
                log.info(f'Now following plan={plan}')
            else:
                log.info(f'Did not find plan')
                if not found_at_least_one_plan and self.abstract_planning and ct < 2:
                    for _ in range(9):
                        cur_obj_list, cur_game_state = self.atari_env.step('NOOP')
                        self.actions_taken.append("NOOP")
                        obj_lists.append(cur_obj_list.deepcopy())
                        game_states.append(cur_game_state)
                        taken_actions.append('NOOP')
                        history_actions.append('NOOP')
                    log.info('Waited for 10 frames to see if waiting helps since we havent found a plan here')
                    continue
                else:
                    break

            for idx, action in enumerate(plan):
                cur_obj_list, cur_game_state = self.atari_env.step(action)
                self.actions_taken.append(action)

                obj_lists.append(cur_obj_list.deepcopy())
                game_states.append(cur_game_state)
                taken_actions.append(action)
                history_actions.append(action)

                # Found target abstract state in a concrete state that is stable
                if self._abstract_state(
                        cur_obj_list, target_id=target_id
                ) == target_abstract_state and (target_id is not None or self._is_stable_state(
                        cur_obj_list, world_model)):
                    break

                # If player dies, does not count
                if cur_game_state == GameState.RESTART:  # player died
                    died = True
                    break

                # Check whether we are on a good path to reaching the target
                # Don't check too often though, it's slow
                if rng.random() < self.config.agent.short_term_accuracy:
                    if not self._on_good_path(cur_obj_list,
                                              plan[idx + 1:],
                                              target_abstract_state,
                                              world_model,
                                              target_id=target_id):
                        log.info(f'Plan failed after taking {taken_actions}')
                        plan = plan[idx + 1:]
                        # TODO: If dies, should give up
                        break

            # Found target abstract state in a concrete state that is stable
            if self._abstract_state(
                    cur_obj_list, target_id=target_id
            ) == target_abstract_state and (target_id is not None or self._is_stable_state(
                    cur_obj_list, world_model)):
                success = True
                log.info(f'Success in getting to {target_abstract_state}!')
                break

            if died:
                log.info(f'Died trying to get to {target_abstract_state}!')
                break
            
            c = [
                StateTransitionTriplet(*data)
                for data in zip(obj_lists[start_index-1:], taken_actions[start_index-1:], obj_lists[start_index-1+1:],
                                game_states[start_index-1:], game_states[start_index-1+1:])
            ]
            cur_c = cur_c + c

            # fast update of world model
            if self.config.agent.fast_world_update and ct % (self.config.agent.max_iter // 10) == 0:
                self.world_learner.update_world_model(
                    cur_c, fast=True, player_only=self.config.agent.update_player_only)
                cur_c = []
            world_model = self.world_learner.world_model

        # This is to see if we die
        for _ in range(1):
            cur_obj_list, cur_game_state = self.atari_env.step('NOOP')
            self.actions_taken.append("NOOP")
            obj_lists.append(cur_obj_list.deepcopy())
            game_states.append(cur_game_state)
            taken_actions.append('NOOP')
            history_actions.append('NOOP')

            # If player dies, does not count
            if cur_game_state == GameState.RESTART:  # player died
                died = True
                break
        if died:
            log.info('Succeeded but actually died')
            success = False
        
        
        # load a snapshot of world_learner saved at the beginning so that fast update doesn't affect us
        if self.config.agent.fast_world_update:
            self.world_learner.load_snapshot()

        if self.config.agent.permanent_world_update and (not success) and (
                not (self.config.agent.ignore_monster and died_by_monster)):
            # Permanently update (with slow learning) world model
            log.info(
                'Since we failed, permanently updates world model based on everything it has seen so far'
            )
            all_c = [
                StateTransitionTriplet(*data)
                for data in zip(obj_lists, taken_actions, obj_lists[1:],
                                game_states, game_states[1:])
            ]
            if found_at_least_one_plan:
                self.world_learner.update_world_model(
                    all_c, fast=False, player_only=self.config.agent.update_player_only)  # This updates the world model forever
                log.info(
                    'DONE -- Since we failed, permanently updates world model based on everything it has seen so far'
                )
            else:
                log.info(
                    'No need to update world model since only one data point')

        log.info(f'Actions taken overall: {history_actions}')
        return cur_obj_list, cur_game_state, success, died_by_monster

    def search_and_prune_in_graph(
        self, cur_abstract_state: str, goal_id: int,
        skills_hsh: Dict[Tuple[str, str],
                         Any], achievables_hsh: Dict[Tuple[str, int], Any]
    ) -> Tuple[Optional[List[str]], Dict[Tuple[str, str], Any], Dict[Tuple[
            str, int], Any]]:
        """
        Search for high-level plan in abstract graph and prune invalid edges.
        Uses BFS to find path from current state to goal state.
        
        Args:
            cur_abstract_state: Current abstract state
            goal_id: Target object ID 
            skills_hsh: Dictionary of available skills/transitions
            achievables_hsh: Dictionary of achievable goal states
        
        Returns:
            Tuple of (plan, updated skills hash, updated achievables hash)
        """
        if self.config.agent.use_ideal_plan:
            # for basic9
            ideal_plan = [
                str([-1, -1, 8, 9]),  # first plaform and first ladder
                str([-1, -1, 10, -1]),  # left conveyer belt
                str([5, -1, -1, -1]),  # rope
                str([-1, -1, 16,-1]),  # right ladder and mid right platform
                str([-1, -1, -1, 14]),  # right ladder
                str([-1, -1, 17, 14]),  # right ladder and low platform:
                str([-1, -1, 17, -1]),  # low platform
                str([-1, -1, 17, 13]),# left ladder and low platform
                str([-1, -1, -1, 13]),  # left ladder and low platform
                str([-1, -1, 15, 13])  # left ladder and low platform
            ]
            return ideal_plan, skills_hsh, achievables_hsh

        # Building skills matrix
        skills_mat = defaultdict(list)  # key as abstract and value as concrete
        for x, y in skills_hsh.keys():
            if x != self.null_abstract_state and y != self.null_abstract_state:
                skills_mat[x].append(y)

        world_model = self.world_learner.world_model

        q = [(cur_abstract_state, [cur_abstract_state])]
        hsh = {cur_abstract_state: True}
        while len(q) > 0:
            abstract_state, visited_abstract_states = q[0]
            q = q[1:]
            if (abstract_state, goal_id) in achievables_hsh:
                cur_obj_list, plan, _ = achievables_hsh[(abstract_state,
                                                         goal_id)]
                # If it is no longer consistent with current world model, delete
                if self.config.agent.prune_bad_edges and not self._on_good_path(
                        cur_obj_list,
                        plan,
                        str([goal_id]),
                        world_model,
                        target_id=goal_id):
                    log.info(
                        f'Deleting achievable edge {abstract_state} --> {goal_id}'
                    )
                    del achievables_hsh[(abstract_state, goal_id)]
                else:
                    return visited_abstract_states, skills_hsh, achievables_hsh

            for abstract_neighbor in skills_mat[abstract_state]:
                if abstract_neighbor not in hsh:
                    cur_obj_list, plan, _ = skills_hsh[(abstract_state,
                                                        abstract_neighbor)]
                    # If it is no longer consistent with current world model, delete
                    if self.config.agent.prune_bad_edges and not self._on_good_path(
                            cur_obj_list, plan, abstract_neighbor,
                            world_model):
                        log.info(
                            f'Deleting skill edge {abstract_state} --> {abstract_neighbor}'
                        )
                        del skills_hsh[(abstract_state, abstract_neighbor)]
                        continue

                    q.append((abstract_neighbor,
                              visited_abstract_states + [abstract_neighbor]))
                    hsh[abstract_neighbor] = True
        return None, skills_hsh, achievables_hsh

    def build_graph(self,
                    obj_list: ObjListWithMemory,
                    n_budget_iterations: int,
                    goal_obj_type: str,
                    load: bool = False) -> Tuple[Dict, Dict]:
        """
        Build or load abstract graph of environment states and transitions.
        Uses parallel MCTS searches to discover possible transitions.

        Args:
            obj_list: Initial object list
            n_budget_iterations: Number of MCTS iterations per search
            load: Whether to load existing graph from disk

        Returns:
            Tuple of (skills hash, achievables hash) defining the graph
        """
        if self.n_build_graph_calls >= 5:
            raise Exception("Too many graph builds")
        self.n_build_graph_calls += 1
        # INITIAL SETUP
        world_model = self.world_learner.world_model
        self.n_abstract_features = len(world_model.constraints)
        self.null_abstract_state = str([-1] * self.n_abstract_features)
        os.makedirs(f'saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}',
                    exist_ok=True)

        # q: Queue of states to explore
        # visited_abs_states: Dictionary tracking which states have been visited
        # skills_hsh: Dictionary storing possible transitions between states
        # achievables_hsh: Dictionary storing states that can be reached
        q, visited_abs_states, skills_hsh, achievables_hsh =\
              self._initialize_graph(obj_list, n_budget_iterations, load)

        log.info("Building graph...")

        # PREPARE WORLD MODEL FOR JOBS (remove callables, measure size)
        no_callables_world_model_path = self._prepare_world_model_for_jobs(
            world_model)

        # SETUP FOR BUILD LOOP
        job_q: List[Tuple] = []
        running_jobs: List[Tuple] = []
        max_running_jobs_ct = 128
        folder = uuid.uuid4()  # Create a unique identifier for this run
        num_iterations = 0
        num_job_finished = 0
        start_time = time.time()
        
        log.info(f'To restart, create a restart_graph.txt file in tmp_params/{folder}')

        # MAIN BUILD LOOP
        while len(q) > 0 or len(job_q) > 0 or len(running_jobs) > 0:
            # Check for restart file
            restart_file = f"tmp_params/{folder}/restart_graph.txt"
            if os.path.exists(restart_file) or time.time() - start_time > 3600 * 5:
                if time.time() - start_time > 3600 * 5:
                    log.info('Restarting graph building process after 5 hours')
                else:
                    log.info("Restarting graph building process after detecting restart file")
                
                # Cancel all SLURM jobs for this folder
                log.info("Cancelling all SLURM jobs for this session...")
                for job_id in self.slurm_job_ids:
                    os.system(f"scancel {job_id}")
                
                # Resubmit the running jobs
                log.info("Resubmitting cancelled jobs...")
                for out_id, args in running_jobs:
                    in_id = uuid.uuid4()
                    in_pkl_path = f"tmp_params/{folder}/in-{in_id}.pickle"
                    out_pkl_path = f"tmp_params/{folder}/{out_id}.pickle"
                    
                    with open(in_pkl_path, "wb") as f:
                        pickle.dump(args, f)
                    
                    if args[0].break_g2:
                        os.chdir("/home/wp237/scripts")
                        job_id = os.popen(f"sbatch --parsable ./mcts.sub {in_pkl_path} {out_pkl_path}").read().strip()
                        self.slurm_job_ids.append(job_id)
                        os.chdir("/home/wp237/active-infer-python-world")
                    else:
                        if self.config.debug_mode:
                            log_path = f"tmp_params/{folder}/{out_id}.log"
                        else:
                            log_path = "/dev/null"
                        job_id = os.popen(f"sbatch --parsable run_mcts.py --in_file {in_pkl_path} --out_file {out_pkl_path} > {log_path} 2>&1").read().strip()
                        self.slurm_job_ids.append(job_id)
                
                # Remove restart file if it exists
                if os.path.exists(restart_file):
                    os.remove(restart_file)
                
                # Reset timer but keep queue and progress
                start_time = time.time()
                continue

            # Step 1: Dequeue states and create new jobs
            job_q = self._create_jobs_from_queue(
                q, job_q, n_budget_iterations, 
                goal_obj_type, no_callables_world_model_path)

            # Step 2: Launch as many jobs as possible
            running_jobs = self._launch_jobs(job_q, running_jobs, folder,
                                             max_running_jobs_ct)

            # Step 3: Collect any finished jobs and update graph
            num_finished, running_jobs = self._collect_finished_jobs(
                running_jobs, folder, skills_hsh, achievables_hsh,
                visited_abs_states, q)
            num_job_finished += num_finished

            # Step 4: Log progress and break if over time
            num_iterations = self._log_progress(num_iterations, q, job_q,
                                                running_jobs, num_job_finished)

        # Optionally save the initial graph if load was True
        if load:
            self.save_graph(n_budget_iterations, q, visited_abs_states,
                            skills_hsh, achievables_hsh)

        log.info("Done building graph")
        return skills_hsh, achievables_hsh

    def _initialize_graph(self, obj_list: ObjListWithMemory,
                          n_budget_iterations: int,
                          load: bool) -> Tuple[List, Dict, Dict, Dict]:
        """Loads or initializes the graph structures (queue and dicts)."""
        if load and self.load_graph(n_budget_iterations):
            q, hsh, skills_hsh, achievables_hsh = self.load_graph(
                n_budget_iterations)
        else:
            abstract_state = self._abstract_state(obj_list)
            q = [(abstract_state, obj_list)]
            hsh = {abstract_state: True}
            skills_hsh = {}
            achievables_hsh = {}
        return q, hsh, skills_hsh, achievables_hsh

    def _prepare_world_model_for_jobs(self, world_model) -> str:
        """Removes callables, saves the stripped model, and logs its size."""
        no_callables_world_model = world_model.remove_callables()
        path = save_world_model_to_path(no_callables_world_model)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        log.info(f"Size of no-callable world_model: {size_mb:.2f} MB")
        return path

    def _create_jobs_from_queue(
            self, q: List[Tuple[str, ObjListWithMemory]], job_q: List[Tuple],
            n_budget_iterations: int,
            goal_obj_type: str,
            no_callables_world_model_path: str) -> List[Tuple]:
        """Dequeues items, logs them, and creates job inputs for each item."""
        while len(q) > 0:
            cur_abstract_state, obj_list = q.pop(0)
            log.info(f"Start searching for edges at {cur_abstract_state}")

            # Achievables: possible_target_ids
            possible_target_ids = [self._get_goal_id(obj_list, goal_obj_type)]
            args1 = [(self.config, n_budget_iterations, obj_list,
                      no_callables_world_model_path, str([t_id]), t_id)
                     for t_id in possible_target_ids]

            # possible_abstract_states
            possible_abstract_states = self._get_possible_abstract_states(
                obj_list)
            args2 = [(self.config, n_budget_iterations, obj_list,
                      no_callables_world_model_path, pas, None)
                     for pas in possible_abstract_states]

            job_q.extend(args1)
            job_q.extend(args2)

        return job_q

    def _launch_jobs(self, job_q: List[Tuple], running_jobs: List[Tuple],
                     folder: uuid.UUID, max_jobs: int) -> List[Tuple]:
        """Launches jobs (up to max_jobs) from job_q in parallel."""
        while job_q and len(running_jobs) <= max_jobs:
            args = job_q.pop(0)
            in_id = uuid.uuid4()
            out_id = uuid.uuid4()
            os.makedirs(f"tmp_params/{folder}", exist_ok=True)

            # Save input pickle
            in_pkl_path = f"tmp_params/{folder}/in-{in_id}.pickle"
            out_pkl_path = f"tmp_params/{folder}/{out_id}.pickle"
            with open(in_pkl_path, "wb") as f:
                pickle.dump(args, f)

            # Decide how to run (slurm vs. python, etc.)
            if args[0].agent.quick_build_graph:
                (_, n_budget_iterations, obj_list, _,
                target_abstract_state, target_id) = args
                
                # self.save_edge_as_gif((obj_list, ['LEFT', 'LEFT', 'LEFT', 'LEFT', 'FIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'FIRE', 'FIRE', 'FIRE', 'FIRE', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'RIGHT', 'LEFT', 'RIGHT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'FIRE', 'LEFT', 'NOOP', 'LEFTFIRE', 'LEFT', 'NOOP', 'FIRE'], None),
                #                       'see.gif',
                #                       self.world_learner.world_model)
                
                plan, concrete_state = self.mcts.search(obj_list,
                                                        target_abstract_state,
                                                        self.world_learner.world_model,
                                                        iterations=n_budget_iterations,
                                                        target_id=target_id,
                                                        ret_concrete_state=True)
                
                with open(out_pkl_path, 'wb') as f:
                    pickle.dump((plan, concrete_state), f)
                
            else:
                if args[0].break_g2:
                    os.chdir("/home/wp237/scripts")
                    # Capture the job ID from sbatch output
                    job_id = os.popen(f"sbatch --parsable ./mcts.sub {in_pkl_path} {out_pkl_path}").read().strip()
                    self.slurm_job_ids.append(job_id)
                    os.chdir("/home/wp237/active-infer-python-world")
                else:
                    if self.config.debug_mode:
                        log.info(f"tmp_params/{folder}/{out_id}.log")
                        log_path = f"tmp_params/{folder}/{out_id}.log"
                    else:
                        log_path = "/dev/null"
                    os.system(f"nohup python run_mcts.py "
                            f"--in_file {in_pkl_path} "
                            f"--out_file {out_pkl_path} "
                            f"> {log_path} 2>&1 &")

            running_jobs.append((out_id, args))

        return running_jobs

    def _collect_finished_jobs(self, running_jobs: List[Tuple],
                               folder: uuid.UUID, skills_hsh: Dict,
                               achievables_hsh: Dict, hsh: Dict,
                               q: List[Tuple[str, ObjListWithMemory]]) -> int:
        """
        Checks which jobs have finished, updates the graph structures,
        and removes those jobs from the running list.
        """
        to_remove_indices = []
        for idx, (out_id, args) in enumerate(running_jobs):
            out_file = f"tmp_params/{folder}/{out_id}.pickle"
            if os.path.exists(out_file):
                try:
                    with open(out_file, "rb") as f:
                        plan, concrete_state = pickle.load(f)
                except Exception as e:
                    continue
                to_remove_indices.append(idx)

                cur_obj_list = args[2]
                cur_state = self._abstract_state(cur_obj_list)

                # If last arg is not None => Achievable
                if args[-1] is not None:
                    target_id = args[-1]
                    if plan is not None:
                        log.info(f"Found achievable {cur_state} "
                                 f"--> {target_id} with {plan}")
                        achievables_hsh[(cur_state, target_id)] = (
                            cur_obj_list.deepcopy(), plan, concrete_state)
                else:
                    # For Skills
                    possible_state = args[-2]
                    if possible_state == cur_state:
                        continue
                    if plan is not None:
                        log.info(f"Found skill {cur_state} "
                                 f"--> {possible_state} with {plan}")
                        skills_hsh[(cur_state, possible_state)] = (
                            cur_obj_list.deepcopy(), plan, concrete_state)
                        if possible_state not in hsh:
                            hsh[possible_state] = True
                            q.append((possible_state, concrete_state))

        # Remove finished jobs
        num_finished_jobs = len(to_remove_indices)
        running_jobs = [
            x for i, x in enumerate(running_jobs) if i not in to_remove_indices
        ]
        return num_finished_jobs, running_jobs

    def _log_progress(self, ct: int, q: List, job_q: List, running_jobs: List,
                      finished_jobs_ct: int) -> int:
        """Logs progress periodically"""
        time.sleep(10)
        ct += 1
        if ct % 6 == 0:
            log.info(
                f"Current progress: q len: {len(q)}; job_q len: {len(job_q)}; "
                f"running_jobs {len(running_jobs)} finished_jobs {finished_jobs_ct}"
            )
        return ct

    def _try_load_graph(self, n_budget_iterations: int) -> bool:
        """Try to load graph from disk"""
        os.makedirs(f'saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}',
                    exist_ok=True)
        return self.load_graph(n_budget_iterations) is not False

    def _prepare_world_model(self, world_model: Any) -> str:
        """Prepare world model for parallel processing"""
        no_callables_model = world_model.remove_callables()
        model_path = save_world_model_to_path(no_callables_model)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        log.info(f"Size of no-callable world_model: {size_mb:.2f} MB")
        return model_path

    def _should_continue_building(self, q: List[Tuple[str, ObjListWithMemory]],
                                  job_queue: List[Any],
                                  running_jobs: List[Any],
                                  start_time: float) -> bool:
        """Check if graph building should continue"""
        work_remaining = len(q) > 0 or len(job_queue) > 0 or len(
            running_jobs) > 0
        time_remaining = time.time() - start_time <= 3600
        return work_remaining and time_remaining

    def _should_log_progress(self, iterations: int) -> bool:
        """Check if progress should be logged"""
        return iterations % 6 == 0

    def _on_good_path(self,
                      obj_list: Any,
                      action_seq: List[str],
                      target_abstract_state: str,
                      world_model: Any,
                      target_id: Optional[int] = None) -> bool:
        """Check if action sequence leads to target state according to world model"""
        cur_obj_list = obj_list.deepcopy()
        success = False
        memory = cur_obj_list.memory
        try:
            for action in action_seq:
                old_obj_list = cur_obj_list
                cur_obj_list = world_model.sample_next_scene(
                    cur_obj_list, action, memory=memory, det=self.config.det_world_model)
                # Check if died
                memory.add_obj_list_and_action(old_obj_list, action)
                player_objs = cur_obj_list.get_objs_by_obj_type('player')
                if len(player_objs) != len(old_obj_list.get_objs_by_obj_type('player')) or (len(player_objs) > 0 and player_objs[0].history['deleted'][-2] == 1):
                    break

                if self._abstract_state(
                        ObjListWithMemory(cur_obj_list, memory), target_id=target_id
                ) == target_abstract_state and (target_id is not None or self._is_stable_state(
                        ObjListWithMemory(cur_obj_list, memory), world_model)):
                    success = True
                    break
        except Exception as e:
            log.info(f"Error on good path: {e}")
            return False
        return success

    def _is_stable_state(self, obj_list: ObjListWithMemory, world_model: Any) -> bool:
        """Check if state is stable (doesn't change under NOOP action)"""
        new_obj_list = obj_list.deepcopy()
        memory = new_obj_list.memory
        new_obj_list = world_model.sample_next_scene(
            new_obj_list, 'NOOP', memory=memory, det=self.config.det_world_model)
        return self._abstract_state(ObjListWithMemory(new_obj_list, memory)) == self._abstract_state(
            obj_list)  # TODO: be careful

    def _abstract_state(self,
                        obj_list: Any,
                        target_id: Optional[int] = None) -> str:
        """Convert concrete state to abstract state representation"""
        if target_id is not None:
            # TODO: don't restrict to player
            try:
                player_obj = obj_list.get_objs_by_obj_type('player')[0]
                target_obj = obj_list.get_obj_by_id(target_id)
            except:
                return '[-1]'
            return str([target_id
                        ]) if player_obj.overlaps(target_obj) else '[-1]'
        else:
            return str(self.world_learner.world_model.get_features(obj_list))

    def _get_possible_abstract_states(self, obj_list: Any) -> List[str]:
        """Generate list of possible abstract states based on current objects"""
        constraints = self.world_learner.world_model.constraints
        all_possibilities = []
        for rule in constraints.rules:
            pattern = r"get_objs_by_obj_type\('([^']+)'\)"

            # Find all matches that are not 'player'
            matches = re.findall(pattern, rule)

            possibilities = [
            ]  # for possible object id that can satisfy this constraint
            for match in matches:
                if match != 'player':
                    possibilities = possibilities + [
                        (obj.id, obj)
                        for obj in obj_list.get_objs_by_obj_type(match)
                    ]
            all_possibilities.append(possibilities)

        possible_abstract_states = []
        for i in range(len(all_possibilities)):
            for id1, obj1 in all_possibilities[i]:
                abstract_state = [-1] * len(all_possibilities)
                abstract_state[i] = id1
                possible_abstract_states.append(str(abstract_state))
                for j in range(i + 1, len(all_possibilities)):
                    for id2, obj2 in all_possibilities[j]:
                        if id1 != id2 and \
                            check_overlap((obj1.left_side, obj1.right_side), (obj2.left_side, obj2.right_side)) and \
                                check_overlap((obj1.top_side, obj1.bottom_side), (obj2.top_side, obj2.bottom_side)):
                            abstract_state = [-1] * len(all_possibilities)
                            abstract_state[i] = id1
                            abstract_state[j] = id2
                            possible_abstract_states.append(
                                str(abstract_state))
        return possible_abstract_states

    def save_edge_as_gif(self, edge, filename, world_model=None):
        """Save visualization of state transition as GIF"""
        if world_model is None:
            world_model = self.world_learner.world_model
        obj_list, action_seq, neighbor = edge
        obj_list = obj_list.deepcopy()
        frames = [self.renderer.render(obj_list)]
        memory = obj_list.memory
        for a in action_seq:
            old_obj_list = obj_list
            obj_list = world_model.sample_next_scene(
                obj_list, a, memory=memory, det=self.config.det_world_model)
            memory.add_obj_list_and_action(old_obj_list, a)
            frames.append(self.renderer.render(obj_list))
        log.info(f'Saving {filename}')
        save_frames_as_gif(
            frames,
            f'./saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/',
            filename)

    def save_abstract_plan_as_gif(self, states, filename, repeat_rate=30):
        """Save visualization of abstract plan as GIF"""
        frames = []
        for state in states:
            frames = frames + [self.renderer.render(state)] * repeat_rate
        log.info(f'Saving {filename}')
        save_frames_as_gif(
            frames,
            f'./saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/',
            filename)

    def save_edges_as_gif(self,
                          edges,
                          filename,
                          repeat_rate=30,
                          world_model=None):
        """Save visualization of multiple transitions as GIF"""
        if world_model is None:
            world_model = self.world_learner.world_model

        frames = []
        for edge in edges:
            obj_list, action_seq, neighbor = edge
            obj_list = obj_list.deepcopy()
            frames = frames + [self.renderer.render(obj_list)] * repeat_rate
            memory = obj_list.memory
            for a in action_seq:
                old_obj_list = obj_list
                obj_list = world_model.sample_next_scene(
                    obj_list, a, memory=memory, det=self.config.det_world_model)
                memory.add_obj_list_and_action(old_obj_list, a)
                frames.append(self.renderer.render(obj_list))
            frames = frames + [self.renderer.render(neighbor)] * repeat_rate
        log.info(f'Saving {filename}')
        save_frames_as_gif(
            frames,
            f'./saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/',
            filename)

    def load_graph(self, n_budget_iterations):
        """Load abstract graph from disk"""
        path = f'saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/'+\
                f'graph_{n_budget_iterations}.pickle'
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            q, hsh, skills_hsh, achievables_hsh = tuple(data)

            log.info(f'Loaded n_node_visited={len(hsh)} ' +
                     f'n_discovered_edges={len(skills_hsh)}')
            return q, hsh, skills_hsh, achievables_hsh
        return False

    def save_graph(self, n_budget_iterations, q, hsh, skills_hsh,
                   achievables_hsh):
        """Save abstract graph to disk"""
        log.info(
            f'Saving n_node_visited={len(hsh)} n_discovered_edges={len(skills_hsh)}'
        )
        os.makedirs(f'saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}',
                    exist_ok=True)
        data = [q, hsh, skills_hsh, achievables_hsh]
        with open(
                f'saved_graph_{"wc_" if self.config.method=="worldcoder" else "" if not self.config.no_constraints else "no-c_"}{self.config.task}{self.config.obs_suffix}{"" if self.config.seed == 0 else f"_s{self.config.seed}"}/graph_{n_budget_iterations}.pickle',
                "wb") as f:
            pickle.dump(data, f)
