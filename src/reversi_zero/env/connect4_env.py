from .base_env import BaseEnv, another_player
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

from .base_env import Player, Winner
from .renju_env import RenjuEnv, RenjuBoard


class Connect4Env(RenjuEnv):
    def __init__(self):
        super().__init__()
        self.concurrent_count = 4

    def create_board(self, board_data=None):
        return Connect4Board(board_data)

    def step(self, action):
        # import sys, traceback
        # traceback.print_stack()

        if action is None:
            self._resigned()
            return self.board, {}

        #action_height = action // self.board.width
        action_width = action % self.board.width

        for j in range(Board.height):
            if self.board.white[Board.height-1-j][action_width] == 0 and self.board.black[Board.height-1-j][action_width] == 0:
                if self.next_player == Player.white:
                    self.board.white[Board.height-1-j][action_width] = 1
                else:
                    self.board.black[Board.height-1-j][action_width] = 1
                break

        self.turn += 1
        self.change_to_next_player()

        self.check_for_fours()

        if self.turn >= self.board.width * self.board.height:
            self._game_over()
            if self.winner is None:
                self.winner = Winner.draw

        return self.board, {}

    def legal_moves(self):
        legal = Board.get_empty_board(Board.height, Board.width)

        for i in range(Board.width):
            for j in range(Board.height):
                if self.board.black[Board.height-1-j][i] == 0 and self.board.white[Board.height-1-j][i] == 0:
                    legal[Board.height-1-j][i] = 1
                    break
        return legal

    def render(self):
        print("\nRound: " + str(self.turn))

        for i in range(5, -1, -1):
            print("\t", end="")
            for j in range(7):
                print("| " + str(self.board[i][j]), end=" ")
            print("|")
        print("\t  _   _   _   _   _   _   _ ")
        print("\t  1   2   3   4   5   6   7 ")

        if self.done:
            print("Game Over!")
            if self.winner == Winner.white:
                print("X is the winner")
            elif self.winner == Winner.black:
                print("O is the winner")
            else:
                print("Game was a draw")


class Connect4Board(RenjuBoard):
    height = 6
    width = 7
    is_symmetry = False
    n_labels = 7
    n_inputs = 42

    @staticmethod
    def to_n_labels(legal_moves):
        legal = [0, 0, 0, 0, 0, 0, 0]
        for i in range(Board.width):
            for j in range(Board.height):
                if legal_moves[j][i] == 1:
                    legal[i] = 1
                    break
        return legal

    @staticmethod
    def to_n_label(x, y):
        return x

    def __init__(self, board_data=None, init_type=0):
        super().__init__(board_data, init_type)
