import enum
from logging import getLogger

from reversi_zero.agent.player import HistoryItem
from reversi_zero.agent.player import ReversiPlayer
from reversi_zero.config import Config
from reversi_zero.env.reversi_env import Board
from reversi_zero.env.reversi_env import Player, ReversiEnv
from reversi_zero.lib.model_helpler import load_best_model_weight, reload_newest_next_generation_model_if_changed
from reversi_zero.play_game.common import load_model

logger = getLogger(__name__)

GameEvent = enum.Enum("GameEvent", "update ai_move over pass")


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.human_color = None
        self.observers = []
        self.env = ReversiEnv().reset()
        self.model = self._load_model()
        self.ai = None  # type: ReversiPlayer
        self.last_evaluation = None
        self.last_history = None  # type: HistoryItem

    def add_observer(self, observer_func):
        self.observers.append(observer_func)

    def notify_all(self, event):
        for ob_func in self.observers:
            ob_func(event)

    def start_game(self, human_is_black):
        self.human_color = Player.black if human_is_black else Player.white
        self.env = ReversiEnv().reset()
        self.ai = ReversiPlayer(self.config, self.model)

    def play_next_turn(self):
        self.notify_all(GameEvent.update)

        if self.over:
            self.notify_all(GameEvent.over)
            return

        if self.next_player != self.human_color:
            self.notify_all(GameEvent.ai_move)

    @property
    def over(self):
        return self.env.done

    @property
    def next_player(self):
        return self.env.next_player

    def stone(self, px, py):
        """left top=(0, 0), right bottom=(7,7)"""
        # pos = int(py * Board.width + px)
        # assert 0 <= pos < Board.width * Board.height
        # bit = 1 << pos
        # if self.env.board.black & bit:
        #     return Player.black
        # elif self.env.board.white & bit:
        #     return Player.white
        # return None

        if self.env.board.black[py][px]:
            return Player.black
        elif self.env.board.white[py][px]:
            return Player.white
        return None

    def available(self, px, py):
        pos = int(py * Board.width + px)
        if pos < 0 or Board.width * Board.height <= pos:
            return False
        legal_moves = self.env.legal_moves()
        #return legal_moves & (1 << pos)
        return legal_moves[py][px]

    def move(self, px, py):
        pos = int(py * Board.width + px)
        assert 0 <= pos < Board.width * Board.height
        if self.next_player != self.human_color:
            return False
        self.env.step(pos)

    def _load_model(self):
        return load_model(self.config)

    def move_by_ai(self):
        if self.next_player == self.human_color:
            return False

        board_data = self.env.get_state_of_next_player()
        action = self.ai.action(board_data)
        self.env.step(action)

        self.last_history = self.ai.ask_thought_about(board_data)
        env2 = ReversiEnv().update(board_data, Player.black)
        key2 = self.ai.counter_key(env2)
        if self.last_history:
            self.last_evaluation = self.last_history.values[self.last_history.action]
            logger.debug(f"evaluation by ai={self.last_evaluation}")

        return self.env.get_action_name(action)

    def find_action_from_move(self, move):
        for i in range(len(Board.labels)):
            if Board.labels[i] == move:
                return i
        return None

    def move_by_human(self):
        board_data = self.env.get_state_of_next_player()
        action = self.ai.action(board_data)
        self.env.step(action)
        return self.env.get_action_name(action)


        import chess
        action = None
        while not action:
            try:
                move = input('\nEnter your move: ')
                if chess.Move.from_uci(move) in self.env.board.board.legal_moves:
                    action = self.find_action_from_move(move)
                else:
                    print("That is NOT a valid move :(.")
                    print('legal moves: {0}'.format(list(self.env.board.board.legal_moves)))
            except Exception as ex:
                print(str(ex))

        self.env.step(action)
        return move
