from .base_env import BaseEnv, another_player
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

from .base_env import Player, Winner


class RenjuEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.board = self.create_board()
        self.concurrent_count = 5

    def create_board(self, board_data=None):
        return RenjuBoard(board_data)

    def reset(self):
        super().reset()
        self.board = self.create_board()
        self.resigned = False
        return self

    def update(self, board_data, next_player=Player.black):
        super().update(board_data, next_player)
        self.board = self.create_board(board_data)
        self.turn = sum(self.board.number_of_black_and_white)
        self.resigned = False
        return self

    def player_turn(self):
        return self.next_player

        if self.turn % 2 == 0:
            assert self.next_player == Player.white
            return Player.white
        else:
            assert self.next_player == Player.black
            return Player.black

    def step(self, action):
        # import sys, traceback
        # traceback.print_stack()

        if action is None:
            self._resigned()
            return self.board, {}

        action_height = action // self.board.width
        action_width = action % self.board.width

        if self.board.white[action_height][action_width] == 0 and self.board.black[action_height][action_width] == 0:
            if self.next_player == Player.white:
                self.board.white[action_height][action_width] = 1
            else:
                self.board.black[action_height][action_width] = 1

        self.turn += 1
        self.change_to_next_player()

        self.check_for_fours()

        if self.turn >= self.board.width * self.board.height:
            self._game_over()
            if self.winner is None:
                self.winner = Winner.draw

        return self.board, {}

    def legal_moves(self):
        # legal = self.board.get_empty_board(self.board.height, self.board.width)
        # for i in range(self.board.width):
        #     for j in range(self.board.height):
        #         if self.board.black[j][i] == 0 and self.board.white[j][i] == 0:
        #             legal[j][i] = 1
        # return legal
        return 1 - (self.board.black + self.board.white)

    def _game_over(self):
        self.done = True

    def check_for_fours(self):
        board_a = self.get_board_as_one()
        for i in range(self.board.height):
            for j in range(self.board.width):
                if board_a[i][j] != 0:
                    # check if a vertical four-in-a-row starts at (i, j)
                    if self.vertical_check(board_a, i, j):
                        self._game_over()
                        return

                    # check if a horizontal four-in-a-row starts at (i, j)
                    if self.horizontal_check(board_a, i, j):
                        self._game_over()
                        return

                    # check if a diagonal (either way) four-in-a-row starts at (i, j)
                    diag_fours = self.diagonal_check(board_a, i, j)
                    if diag_fours:
                        self._game_over()
                        return

    def vertical_check(self, board_a, row, col):
        four_in_a_row = False
        consecutive_count = 0

        for i in range(row, self.board.height):
            if board_a[i][col] == board_a[row][col]:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= self.concurrent_count:
            four_in_a_row = True
            if 2 == board_a[row][col]:
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return four_in_a_row

    def horizontal_check(self, board_a, row, col):
        four_in_a_row = False
        consecutive_count = 0

        for j in range(col, self.board.width):
            if board_a[row][j] == board_a[row][col]:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= self.concurrent_count:
            four_in_a_row = True
            if 2 == board_a[row][col]:
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return four_in_a_row

    def diagonal_check(self, board_a, row, col):
        four_in_a_row = False
        count = 0

        consecutive_count = 0
        j = col
        for i in range(row, self.board.height):
            if j >= self.board.width:
                break
            elif board_a[i][j] == board_a[row][col]:
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= self.concurrent_count:
            count += 1
            if 2 == board_a[row][col]:
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        consecutive_count = 0
        j = col
        for i in range(row, -1, -1):
            if j >= self.board.width:
                break
            elif board_a[i][j] == board_a[row][col]:
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= self.concurrent_count:
            count += 1
            if 2 == board_a[row][col]:
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        if count > 0:
            four_in_a_row = True

        return four_in_a_row

    def _resigned(self):
        self._win_another_player()
        self._game_over()

    def _win_another_player(self):
        win_player = another_player(self.next_player)  # type: Player
        if win_player == Player.black:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def get_board_as_one(self):
        # a = []
        # for j in range(self.board.height):
        #     a.append([])
        #     for i in range(self.board.width):
        #         if self.board.black[j][i]:
        #             a[j].append('O')
        #         elif self.board.white[j][i]:
        #             a[j].append('X')
        #         else:
        #             a[j].append(' ')
        # return a
        return self.board.black + self.board.white * 2

    def get_board_data(self, reverse=False):
        return self.board.get_data(reverse)

    def get_state_of_next_player(self):
        if self.next_player == Player.black:
            own, enemy = self.board.black, self.board.white
        elif self.next_player == Player.white:
            own, enemy = self.board.white, self.board.black
        return (np.copy(own), np.copy(enemy))

    # array to long
    def get_hashable_board_data(self):
        # r = []
        # for j in range(self.board.height):
        #     r1 = []
        #     for i in range(self.board.width):
        #         if board[j][i]:
        #             r1.append('1')
        #         # elif white[j][i]:
        #         #     r1.append('X')
        #         else:
        #             r1.append(' ')
        #     r.append(''.join(r1))
        # return ''.join(r)

        #return (self.board.get_hashable_board_data(self.board.black), self.board.get_hashable_board_data(self.board.white))
        return ' '.join(map(lambda x: str(x), self.board.black.flatten().tolist())), ' '.join(map(lambda x: str(x), self.board.white.flatten().tolist()))
        #return tuple(np.ndarray.flatten(self.board.black)), tuple(np.ndarray.flatten(self.board.white))

    # string to array(black,white)
    #@staticmethod
    def get_antihash_board_data(self, x):
        # r = [None, None]
        # for k in range(2):
        #     r[k] = self.board.get_empty_board(self.board.height, self.board.width)
        #     for j in range(self.board.height):
        #         for i in range(self.board.width):
        #             if x[k][j * self.board.width + i] == ' ':
        #                 r[k][j][i] = 0
        #             else:
        #                 r[k][j][i] = 1
        return (np.reshape(np.array(list(map(lambda x: int(x), x[0].split(' ')))), (self.board.height, self.board.width)), np.reshape(np.array(list(map(lambda x: int(x), x[1].split(' ')))), (self.board.height, self.board.width)))
        #return np.reshape(np.array(x[0]), (self.board.height, self.board.width)), np.reshape(np.array(x[1]), (self.board.height, self.board.width))

    def flip_vertical(self):
        self.board.black = np.flipud(self.board.black)
        self.board.white = np.flipud(self.board.white)
        return self

    def rotate90(self):
        self.board.black = np.rot90(self.board.black, k=1)
        self.board.white = np.rot90(self.board.white, k=1)
        return self

    def get_game_result(self):
        if self.winner == Winner.white:
            mes = "white wins"
        elif self.winner == Winner.black:
            mes = "black wins"
        else:
            mes = "draw"
        return mes

    def get_action_name(self, action):
        return '{0}, {1}'.format(action // self.board.width, action % self.board.width)


class RenjuBoard:
    height = 8
    width = 8
    is_symmetry = False
    n_labels = 64
    n_inputs = 64

    # @staticmethod
    # def flip_vertical(board):
    #     r = [[0 for i in range(self.board.width)] for j in range(self.board.height)]
    #     for i in range(self.board.height):
    #         for j in range(self.board.width):
    #             r[j][i] = board[j][self.board.height-1-i]
    #     return r

    # @staticmethod
    # def rotate90(board):
    #     r = [[0 for i in range(self.board.width)] for j in range(self.board.height)]
    #     for i in range(self.board.height):
    #         for j in range(self.board.width):
    #             r[j][i] = board[self.board.height-1-i][j]
    #     return r

    # @staticmethod
    # def board_to_string(black, white):
    #     r = []
    #     for j in range(self.height):
    #         r1 = []
    #         for i in range(self.width):
    #             if black[j][i]:
    #                 r1.append('O')
    #             elif white[j][i]:
    #                 r1.append('X')
    #             else:
    #                 r1.append('.')
    #         r.append(''.join(r1))
    #     return '\n'.join(r)


    @staticmethod
    def to_n_labels(legal_moves):
        legal_moves_arr = np.array(legal_moves, dtype=np.uint8)
        legal_moves_arr = legal_moves_arr.reshape(legal_moves_arr.size)
        return legal_moves_arr

    @staticmethod
    def to_n_label(x, y):
        return y * RenjuBoard.width+x

    @staticmethod
    def get_empty_board(height, width):
        return np.zeros([height, width], dtype=np.uint8)

    def __init__(self, board_data=None, init_type=0):
        if board_data:
            self.black = np.copy(board_data[0])
            self.white = np.copy(board_data[1])
        else:
            self.black = self.get_empty_board(self.height, self.width)
            self.white = self.get_empty_board(self.height, self.width)

        if init_type:
            self.black, self.white = self.white, self.black

    @property
    def number_of_black_and_white(self):
        #return sum(len(list(filter(lambda x: x, self.black[i]))) for i in range(self.height)), sum(len(list(filter(lambda x: x, self.white[i]))) for i in range(self.height))
        return sum(self.black) + sum(self.white)

    def get_data(self, reverse=False):
        if not reverse:
            return (self.black, self.white)
        else:
            return (self.white, self.black)
