import logging
import hydra

from learners.obj_model_learner import ObjModelLearner
from data.atari import load_atari_observations
from classes.helper import set_global_constants, StateTransitionTriplet

log = logging.getLogger('main')
log.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    if config.database_path is None:
        config.database_path = f'completions_atari_{config.task.lower()}.db'
    
    if config.world_model_learner.obj_type != 'all':
        obj_model_learner = ObjModelLearner(
            config, config.world_model_learner.obj_type, False, [], [], [], [])
        load_success = obj_model_learner.load(None)

        if not load_success:
            raise Exception('Load failed')

        obj_model_learner.display_rules('non_creation')
        obj_model_learner.display_rules('creation')
        obj_model_learner.display_rules('constraints')
    else:
        set_global_constants(config.task)

        # Load observations
        observations, actions, game_states = load_atari_observations(
            config.task + config.obs_suffix)

        # Optional: Use subset of observations
        if config.obs_index != -1:
            observations = observations[config.obs_index:config.obs_index +
                                        config.obs_index_length + 1]
            actions = actions[config.obs_index:config.obs_index +
                              config.obs_index_length]
            game_states = game_states[config.obs_index:config.obs_index +
                                      config.obs_index_length + 1]

        obj_type_dict = {}
        for obs in observations:
            for obj in obs:
                if config.world_model_learner.exclude_score_objects and \
                        obj.obj_type in ['playerscore', 'enemyscore', 'score', 'timer', 'lifecount', 'life']:
                    continue
                obj_type_dict[obj.obj_type] = True
        obj_types = list(obj_type_dict)

        total_lines = 0

        for obj_type in obj_types:
            # log.info('Object type: {}'.format(obj_type))
            print(f'---------- Object type: {obj_type} ----------')
            obj_model_learner = ObjModelLearner(config, obj_type, False, [],
                                                [], [], [])
            load_success = obj_model_learner.load(None)

            if not load_success:
                raise Exception('Load failed')
            
            obj_model_learner.display_rules('non_creation')
            obj_model_learner.display_rules('creation')
            obj_model_learner.display_rules('constraints')
            
            # total_lines += obj_model_learner.count_lines('non_creation')
            # total_lines += obj_model_learner.count_lines('creation')
            # total_lines += obj_model_learner.count_lines('constraints')

            # log.info(
            #     f'Non creation: {len(obj_model_learner.moe_non_creation.rules)}'
            # )
            # log.info(f'Creation: {len(obj_model_learner.moe_creation.rules)}')

            # observations, actions, game_states = load_atari_observations(
            #     config.task + config.obs_suffix)

            # # Optional: Use subset of observations
            # if config.obs_index != -1:
            #     observations = observations[config.obs_index:config.obs_index +
            #                                 config.obs_index_length + 1]
            #     actions = actions[config.obs_index:config.obs_index +
            #                       config.obs_index_length]
            #     game_states = game_states[config.obs_index:config.obs_index +
            #                               config.obs_index_length + 1]

            # for i in range(len(actions)):
            #     x = StateTransitionTriplet(observations[i],
            #                                actions[i],
            #                                observations[i + 1],
            #                                input_game_state=game_states[i],
            #                                output_game_state=game_states[i +
            #                                                              1])
            #     obj_model_learner.add_datapoint(x)
            # for i in range(5, 10):
            #     log.info(i)
            #     for o in x.input_state:
            #         log.info(o.str_w_id())
            #     log.info(x.event)
            #     for o in x.output_state:
            #         log.info(o.str_w_id())

            #     log.info(obj_model_learner._explain_well(i, num=True))

        log.info(f'Total lines: {total_lines}')

if __name__ == '__main__':
    main()
