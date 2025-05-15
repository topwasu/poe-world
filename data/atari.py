from typing import Tuple, List
import pickle

from classes.helper import ObjList, GameState


def load_atari_observations(
        identifier: str) -> Tuple[List[ObjList], List[str], List[GameState]]:
    filename = f"saved_data/obs_{identifier}.pickle"
    with open(filename, "rb") as f:
        observations, actions, game_states = pickle.load(f)
    return observations, actions, game_states
