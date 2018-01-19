import cchess
from .base_env import BaseEnv
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

from .base_env import Player, Winner


class CChessEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        self.board = Board()
        self.resigned = False
        return self

    def update(self, board_data, next_player=Player.black):
        super().update(board_data, next_player)
        self.board = Board(board_data)
        self.turn = 0
        self.resigned = False
        return self

    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action is None:
            self._resigned()
            return self.board, {}

        self.board.board.move_iccs(Board.labels[action])

        self.turn += 1
        self.change_to_next_player()

        if self.board.board.is_game_over() or self.board.board.can_claim_threefold_repetition():
            self._game_over()

        return self.board, {}

    def legal_moves(self):
        self.move_lookup = {k:v for k,v in zip((chess.Move.from_iccs(move) for move in Board.labels), range(len(Board.labels)))}
        legal_moves = [self.move_lookup[mov] for mov in self.board.board.legal_moves]
        return legal_moves

    def _game_over(self):
        self.done = True
        if self.winner is None:
            result = self.board.board.result()
            if result == '1-0':
                self.winner = Winner.white
            elif result == '0-1':
                self.winner = Winner.black
            else:
                val_black, val_white = self.score_board()
                if val_black > val_white:
                    self.winner = Winner.black
                elif val_black < val_white:
                    self.winner = Winner.white
                else:
                    self.winner = Winner.draw

    def score_current(self):
        val_black, val_white = self.score_board()
        if self.board.board.turn == chess.WHITE:
            return val_black - val_white
        else:
            return val_white - val_black

    def score_board(self):
        board_state = Board.replace_tags(self.board)
        pieces_white = [val if val.isupper() and val != "1" else 0 for val in board_state.split(" ")[0]]
        pieces_black = [val if val.islower() and val != "1" else 0 for val in board_state.split(" ")[0]]
        val_white = 0.0
        val_black = 0.0
        for piece in pieces_white:
            if piece == 'Q':
                val_white += 10.0
            elif piece == 'R':
                val_white += 5.5
            elif piece == 'B':
                val_white += 3.5
            elif piece == 'N':
                val_white += 3
            elif piece == 'P':
                val_white += 1
        for piece in pieces_black:
            if piece == 'q':
                val_black += 10.0
            elif piece == 'r':
                val_black += 5.5
            elif piece == 'b':
                val_black += 3.5
            elif piece == 'n':
                val_black += 3
            elif piece == 'p':
                val_black += 1
        return val_black, val_white

    def _resigned(self):
        self._win_another_player()
        self._game_over()
        self.resigned = True

    def _win_another_player(self):
        if self.board.board.turn == chess.BLACK:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    # @property
    # def observation(self):
    #     return self.board.fen()

    def black_and_white_plane(self):
        return Board.black_and_white_plane(self.board)

    def get_state_of_next_player(self):
        return self.board.board.fen()

    # array to long
    def get_hashable_board_data(self):
        return self.board.board.fen()

    def flip_vertical(self):
        raise Exception('')

    def rotate90(self):
        raise Exception('')

    def get_game_result(self):
        if self.winner == Winner.white:
            mes = "X wins"
        elif self.winner == Winner.black:
            mes = "O wins"
        return mes


def create_ucci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(0, 9)] + \
                           [(l1, t) for t in range(0, 9)] + \
                           [(l1 + t, n1 + t) for t in range(-8, 9)] + \
                           [(l1 + t, n1 - t) for t in range(-8, 9)] + \
                           [(l1 + a, n1 + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]] + \
                           [(l1 + a, n1 + b) for (a, b) in [(-2, -2), (-2, 2), (2, -2), (2, 2)]] + \
                           [(l1 + a, n1 + b) for (a, b) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(0, 8) and n2 in range(0, 8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    return labels_array


class Board():
    height = 9
    width = 9
    is_symmetry = False
    n_labels = 1968
    n_inputs = 81
    labels = create_ucci_labels()

    @staticmethod
    def black_and_white_plane(chess_board):
        board_state = Board.replace_tags(chess_board)
        board_white = [ord(val) if val.isupper() and val != "1" else 0 for val in board_state.split(" ")[0]]
        board_white = np.reshape(board_white, (8, 8))
        # Only black plane
        board_black = [ord(val) if val.islower() and val != "1" else 0 for val in board_state.split(" ")[0]]
        board_black = np.reshape(board_black, (8, 8))

        return board_white, board_black

    @staticmethod
    def replace_tags(chess_board):
        board_san = chess_board.board.fen()
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")

        return board_san.replace("/", "")

    # fen -> string
    @staticmethod
    def get_hashable_board(board):
        return board

    # string -> fen
    @staticmethod
    def get_antihash_board(x):
        return x

    @staticmethod
    def to_n_labels(legal_moves):
        legal_labels = np.zeros(len(Board.labels))
        #logger.debug(legal_moves)
        legal_labels[legal_moves] = 1
        return legal_labels

    @staticmethod
    def to_n_label(x, y):
        raise Exception('not supported')

    def __init__(self, board_data=None, init_type=0):
        if board_data:
            self.board = chess.Board(board_data)
        else:
            self.board = chess.Board()

        if init_type:
            raise Exception('')

    def get_data(self, reverse=False):
        return self.board.fen()
