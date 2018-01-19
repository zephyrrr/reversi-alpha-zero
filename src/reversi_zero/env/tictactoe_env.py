from .base_env import BaseEnv, another_player
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

from .base_env import Player, Winner


from .renju_env import RenjuEnv, RenjuBoard


class TicTacToeEnv(RenjuEnv):
    def __init__(self):
        super().__init__()
        self.concurrent_count = 3

    def create_board(self, board_data=None):
        return TicTacToeBoard(board_data)


class TicTacToeBoard(RenjuBoard):
    height = 3
    width = 3
    is_symmetry = False
    n_labels = 9
    n_inputs = 9

    def __init__(self, board_data=None, init_type=0):
        super().__init__(board_data, init_type)
