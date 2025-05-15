from enum import Enum
from abc import ABC, abstractmethod


class BaseActions(str, Enum):
    """Base class for game actions that all games should inherit from."""
    @abstractmethod
    def get_all_possible_actions(cls):
        """Return a list of all possible actions for the game.
        
        Must be implemented by each game-specific action class.
        """
        pass
