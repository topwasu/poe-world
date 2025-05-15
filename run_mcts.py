import argparse
import dill

from agents.mcts import MCTS
from classes.helper import set_global_constants
import pickle


def read_world_model_from_path(path):
    with open(path, 'rb') as f:
        world_model = dill.load(f)
    return world_model


def main():
    parser = argparse.ArgumentParser(
        description='Run MCTS with specified configuration.')
    parser.add_argument('--in_file',
                        type=str,
                        required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--out_file',
                        type=str,
                        required=True,
                        help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.in_file, 'rb') as f:
        params = pickle.load(f)

    (config, n_budget_iterations, obj_list, no_callables_world_model_path,
     target_abstract_state, target_id) = params

    set_global_constants(config.task)
    mcts = MCTS(config)
    no_callables_world_model = read_world_model_from_path(
        no_callables_world_model_path)
    no_callables_world_model.prepare_callables()
    world_model = no_callables_world_model

    plan, concrete_state = mcts.search(obj_list,
                                       target_abstract_state,
                                       world_model,
                                       iterations=n_budget_iterations,
                                       target_id=target_id,
                                       ret_concrete_state=True)

    with open(args.out_file, 'wb') as f:
        pickle.dump((plan, concrete_state), f)


if __name__ == '__main__':
    main()
