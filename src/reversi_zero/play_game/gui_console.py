from logging import getLogger


from reversi_zero.config import Config, PlayWithHumanConfig
from reversi_zero.play_game.game_model import PlayWithHuman
from reversi_zero.env.connect4_env import Connect4Env, Player, Winner
from random import random
from reversi_zero.lib import tf_util

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.2)
    PlayWithHumanConfig().update_play_config(config.play)
    reversi_model = PlayWithHuman(config)

    while True:
        human_is_black = random() < 0.5
        reversi_model.start_game(human_is_black)

        while not reversi_model.env.done:
            reversi_model.env.render()
            if reversi_model.env.next_player == Player.black:
                if not human_is_black:
                    action = reversi_model.move_by_ai()
                    print("IA moves to: " + str(action))
                else:
                    action = reversi_model.move_by_human()
                    print("You move to: " + str(action))
            else:
                if human_is_black:
                    action = reversi_model.move_by_ai()
                    print("IA moves to: " + str(action))
                else:
                    action = reversi_model.move_by_human()
                    print("You move to: " + str(action))

        print("\nEnd of the game.")
        print("Game result:")
        if reversi_model.env.winner == Winner.white:
            print("white({0})wins".format('ai' if human_is_black else 'you'))
        elif env.winner == Winner.black:
            print("black({0})wins".format('you' if human_is_black else 'ai'))
        else:
            print("Game was a draw")
