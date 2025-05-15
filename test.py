from agents.agent import Agent
from agents.mcts import MCTS
from classes.helper import ObjListWithMemory

import logging
log = logging.getLogger(__name__)




class TestAgent(Agent):
    def quick_test(self, cur_obj_list, cur_game_state, goal_obj_type):
        goal_id = self._get_goal_id(cur_obj_list, goal_obj_type)
        if goal_id == -1:
            return None
        
        ideal_plan = [
            str([-1, -1, 8, 9]),  # first plaform and first ladder
            str([-1, 11, -1, -1]),  # left conveyer belt
            str([5, -1, -1, -1]),  # rope
            str([-1, -1, 16,-1]),  # right ladder and mid right platform
            str([-1, -1, -1, 14]),  # right ladder
            str([-1, -1, 17, 14]),  # right ladder and low platform:
            str([-1, -1, 17, -1]),  # low platform
            # str([-1, -1, 17, 13]),# left ladder and low platform
            str([-1, -1, -1, 13]),  # left ladder and low platform
            str([-1, -1, 15, 13])  # left ladder and low platform
        ]

        world_model = self.world_learner.world_model
        
        # memory = cur_obj_list.memory
        # for action in ['DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN'] + ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP'] + ['RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'RIGHTFIRE', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP', 'NOOP'] + ['DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN'] + ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT']:
        #     old_obj_list = cur_obj_list
        #     cur_obj_list = world_model.sample_next_scene(
        #         cur_obj_list, action, memory=memory, det=self.config.det_world_model)
        #     # Check if died
        #     memory.add_obj_list_and_action(old_obj_list, action)
            
        # cur_obj_list = ObjListWithMemory(cur_obj_list, memory)
        
        for target_abstract_state in ideal_plan[1:]:
            log.info(f'Searching for plan to {target_abstract_state}')
            plan, concrete_state = self.mcts.search(cur_obj_list,
                                    target_abstract_state,
                                    world_model,
                                    iterations=1000,
                                    target_id=None,
                                    ret_concrete_state=True)
            if plan is None:
                raise Exception(f'No plan found for {target_abstract_state}')
            
            log.info(f'Got to {target_abstract_state} with plan {plan}')
            cur_obj_list = concrete_state