from .base_env import BaseEnv, another_player
import enum
import numpy as np
from logging import getLogger

from reversi_zero.lib.bitboard import board_to_string, calc_flip, bit_count, find_correct_moves, bit_to_array

logger = getLogger(__name__)
from .base_env import Player, Winner


class ReversiEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        self.board = ReversiBoard()
        return self

    def update(self, board_data, next_player=Player.black):
        super().update(board_data, next_player)
        self.board = ReversiBoard(board_data)
        self.turn = sum(self.board.number_of_black_and_white) - 4
        return self

    def step(self, action):
        """

        :param int|None action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right), None is resign
        :return:
        """
        assert action is None or 0 <= action <= 63, f"Illegal action={action}"

        if action is None:
            self._resigned()
            return self.board, {}

        own2, enemy2 = self.get_own_and_enemy()
        own, enemy = ReversiBoard.board_to_bit(own2), ReversiBoard.board_to_bit(enemy2)
        flipped = calc_flip(action, own, enemy)

        if bit_count(flipped) == 0:
            self.illegal_move_to_lose(action)
            return self.board, {}
        own ^= flipped
        own |= 1 << action
        enemy ^= flipped

        self.set_own_and_enemy(own, enemy)
        self.turn += 1

        if bit_count(find_correct_moves(enemy, own)) > 0:  # there are legal moves for enemy.
            self.change_to_next_player()
        elif bit_count(find_correct_moves(own, enemy)) > 0:  # there are legal moves for me but enemy.
            pass
        else:  # there is no legal moves for me and enemy.
            self._game_over()

        return self.board, {}

    def legal_moves(self):
        own, enemy = ReversiBoard.board_to_bit(self.board.black), ReversiBoard.board_to_bit(self.board.white)
        if self.next_player == Player.white:
            own, enemy = enemy, own
        x = find_correct_moves(own, enemy)
        return ReversiBoard.bit_to_board(x)

    def _game_over(self):
        self.done = True
        if self.winner is None:
            black_num, white_num = self.board.number_of_black_and_white
            if black_num > white_num:
                self.winner = Winner.black
            elif black_num < white_num:
                self.winner = Winner.white
            else:
                self.winner = Winner.draw

    def illegal_move_to_lose(self, action):
        logger.warning(f"Illegal action={action}, No Flipped!")
        self._win_another_player()
        self._game_over()

    def _resigned(self):
        self._win_another_player()
        self._game_over()

    def _win_another_player(self):
        win_player = another_player(self.next_player)  # type: Player
        if win_player == Player.black:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def get_own_and_enemy(self):
        if self.next_player == Player.black:
            own, enemy = self.board.black, self.board.white
        else:
            own, enemy = self.board.white, self.board.black
        return own, enemy

    def set_own_and_enemy(self, own, enemy):
        if self.next_player == Player.black:
            self.board.black, self.board.white = ReversiBoard.bit_to_board(own), ReversiBoard.bit_to_board(enemy)
        else:
            self.board.white, self.board.black = ReversiBoard.bit_to_board(own), ReversiBoard.bit_to_board(enemy)

    def render(self):
        b, w = self.board.number_of_black_and_white
        print(f"next={self.next_player.name} turn={self.turn} B={b} W={w}")
        print(self.board_to_string())

    def board_to_string(self):
        return Board.board_to_string(self.board.black, self.board.white)

    def black_and_white_plane(self):
        return np.array(self.board.black), np.array(self.board.white)

    def get_state_of_next_player(self):
        if self.next_player == Player.black:
            own, enemy = self.board.black, self.board.white
        elif self.next_player == Player.white:
            own, enemy = self.board.white, self.board.black
        return (own, enemy)

    def get_board_data(self, reverse=False):
        return self.board.get_data(reverse)

    # array to long
    def get_hashable_board_data(self):
        return (ReversiBoard.board_to_bit(self.board.black), ReversiBoard.board_to_bit(self.board.white))

    def flip_vertical(self):
        self.board.black = ReversiBoard.flip_vertical(self.board.black)
        self.board.white = ReversiBoard.flip_vertical(self.board.white)
        return self

    def rotate90(self):
        self.board.black = ReversiBoard.rotate90(self.board.black)
        self.board.white = ReversiBoard.rotate90(self.board.white)
        return self

    def get_game_result(self):
        black, white = self.board.number_of_black_and_white
        mes = "black: %d\nwhite: %d\n" % (black, white)
        if black == white:
            mes += "** draw **"
        else:
            mes += "winner: %s" % ["black", "white"][black < white]
        return mes


class ReversiBoard:
    height = 8
    width = 8
    is_symmetry = True
    n_labels = 64
    n_inputs = 64

    @staticmethod
    def flip_vertical(board):
        # r = [[0 for i in range(Board.width)] for j in range(Board.height)]
        # for i in range(Board.height):
        #     for j in range(Board.width):
        #         r[j][i] = board[j][Board.height-1-i]
        # return r
        from reversi_zero.lib.bitboard import flip_vertical
        x = ReversiBoard.board_to_bit(board)
        y = flip_vertical(x)
        return ReversiBoard.bit_to_board(y)

    @staticmethod
    def rotate90(board):
        from reversi_zero.lib.bitboard import rotate90
        x = ReversiBoard.board_to_bit(board)
        y = rotate90(x)
        return ReversiBoard.bit_to_board(y)

    @staticmethod
    def board_to_string(black, white):
        r = []
        for j in range(ReversiBoard.height):
            r1 = []
            for i in range(ReversiBoard.width):
                if black[j][i]:
                    r1.append('O')
                elif white[j][i]:
                    r1.append('X')
                else:
                    r1.append('.')
            r.append(''.join(r1))
        return '\n'.join(r)

    @staticmethod
    def board_to_bit(black):
        r = 0
        for j in range(ReversiBoard.height):
            for i in range(ReversiBoard.width):
                if black[j][i]:
                    r += 1 << (j * ReversiBoard.width + i)
        return r

    @staticmethod
    def bit_to_board(x):
        y = bit_to_array(x, ReversiBoard.n_inputs)
        r = [[0 for i in range(ReversiBoard.width)] for j in range(ReversiBoard.height)]
        for j in range(ReversiBoard.height):
            for i in range(ReversiBoard.width):
                r[j][i] = y[ReversiBoard.width * j + i]
        return r

    # array to long
    @staticmethod
    def get_hashable_board_data(board):
        return ReversiBoard.board_to_bit(board)

    # long to array
    @staticmethod
    def get_antihash_board(x):
        assert type(x) is list
        r = [None, None]
        for k in range(2):
            r[k] = Board.bit_to_board(x[i])
        return (r[0], r[1])

    @staticmethod
    def to_n_labels(legal_moves):
        legal_moves_arr = np.array(legal_moves, dtype=np.uint8)
        legal_moves_arr = legal_moves_arr.reshape(legal_moves_arr.size)
        return legal_moves_arr

    @staticmethod
    def to_n_label(x, y):
        return y*ReversiBoard.width+x

    @staticmethod
    def get_empty_board(height, width):
        return [[0 for i in range(width)] for j in range(height)]

    def __init__(self, board_data=None, init_type=0):
        if board_data:
            if type(board_data[0]) is int:
                self.black = self.get_antihash_board(board_data[0])
                self.white = self.get_antihash_board(board_data[1])
            else:
                self.black = np.copy(board_data[0])
                self.white = np.copy(board_data[1])
        else:
            self.black = self.get_empty_board(self.height, self.width)
            self.white = self.get_empty_board(self.height, self.width)
            self.white[3][3] = 1
            self.white[4][4] = 1
            self.black[3][4] = 1
            self.black[4][3] = 1

        if init_type:
            self.black, self.white = self.white, self.black

    @property
    def number_of_black_and_white(self):
        return sum(len(list(filter(lambda x: x, self.black[i]))) for i in range(self.height)), sum(len(list(filter(lambda x: x, self.white[i]))) for i in range(self.height))

    def get_data(self, reverse=False):
        if not reverse:
            return (self.black, self.white)
        else:
            return (self.white, self.black)
