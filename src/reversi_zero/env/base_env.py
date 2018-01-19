import enum

from logging import getLogger

logger = getLogger(__name__)
# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")
# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


def another_player(player):
    return Player.white if player == Player.black else Player.black


class BaseEnv:
    def __init__(self):
        self.board = None
        self.next_player = None  # type: Player
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner

    def reset(self):
        self.next_player = Player.black
        self.turn = 0
        self.done = False
        self.winner = None
        return self

    def update(self, board_data, next_player=Player.black):
        self.next_player = next_player
        self.done = False
        self.winner = None
        return self

    def step(self, action):
        return self.board, {}

    def change_to_next_player(self):
        self.next_player = another_player(self.next_player)

    @property
    def observation(self):
        return self.board

    def get_action_name(self, action):
        return '{0}, {1}'.format(action // self.board.width, action % self.board.width)
