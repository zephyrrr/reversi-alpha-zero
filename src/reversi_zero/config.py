import os

from moke_config import ConfigBase


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir(config_type):
    return os.path.join(_project_dir(), "data_" + config_type)


class Config(ConfigBase):
    def __init__(self):
        self.type = "default"
        self.opts = Options()
        self.resource = ResourceConfig(config_type)
        self.gui = GuiConfig()
        self.nboard = NBoardConfig()

        self.config_type = config_type
        if config_type == "mini":
            import reversi_zero.configs.mini_reversi as c
        elif config_type == "normal":
            import reversi_zero.configs.normal_reversi as c
        elif config_type == "tictactoe":
            import reversi_zero.configs.normal_tictactoe as c
        elif config_type == "connect4":
            import reversi_zero.configs.normal_connect4 as c
        elif config_type == "renju":
            import reversi_zero.configs.normal_renju as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = ModelConfig()
        self.play = PlayConfig()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()
        self.eval = EvaluateConfig()
        self.play_with_human = PlayWithHumanConfig()


        create_env_py(config_type)


def create_env_py(config_type):
    slist = []
    slist.append('from .base_env import Player, Winner')
    if config_type == "mini" or config_type == "normal":
        slist.append('from .reversi_env2 import ReversiEnv as ReversiEnv')
        slist.append('from .reversi_env2 import ReversiBoard as Board')
    elif config_type == "tictactoe":
        slist.append('from .tictactoe_env import TicTacToeEnv as ReversiEnv')
        slist.append('from .tictactoe_env import TicTacToeBoard as Board')
    elif config_type == "connect4":
        slist.append('from .connect4_env import Connect4Env as ReversiEnv')
        slist.append('from .connect4_env import Connect4Board as Board')
    elif config_type == "renju":
        slist.append('from .renju_env import RenjuEnv as ReversiEnv')
        slist.append('from .renju_env import RenjuBoard as Board')
    else:
        raise RuntimeError(f"unknown config_type: {config_type}")
    fw = open(os.path.join(_project_dir(), "src/reversi_zero/env/reversi_env.py"), 'w')
    for i in slist:
        fw.write(i)
        fw.write('\n')


class Options(ConfigBase):
    new = False


class ResourceConfig(ConfigBase):
    def __init__(self, config_type=''):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir(config_type))
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.self_play_ggf_data_dir = os.path.join(self.data_dir, "self_play-ggf")
        self.ggf_filename_tmpl = "self_play-%s.ggf"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard')
        self.self_play_log_dir = os.path.join(self.tensorboard_log_dir, "self_play")
        self.force_learing_rate_file = os.path.join(self.data_dir, ".force-lr")
        self.force_simulation_num_file = os.path.join(self.data_dir, ".force-sim")
        self.self_play_game_idx_file = os.path.join(self.data_dir, ".self-play-game-idx")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir, self.self_play_log_dir, self.self_play_ggf_data_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class GuiConfig(ConfigBase):
    def __init__(self):
        self.window_size = (400, 440)
        self.window_title = "reversi-alpha-zero-generic"


class PlayWithHumanConfig(ConfigBase):
    def __init__(self):
        self.parallel_search_num = 8
        self.noise_eps = 0
        self.change_tau_turn = 0
        self.resign_threshold = None
        self.use_newest_next_generation_model = True

    def update_play_config(self, pc):
        """

        :param PlayConfig pc:
        :return:
        """
        pc.noise_eps = self.noise_eps
        pc.change_tau_turn = self.change_tau_turn
        pc.parallel_search_num = self.parallel_search_num
        pc.resign_threshold = self.resign_threshold
        pc.use_newest_next_generation_model = self.use_newest_next_generation_model


class NBoardConfig(ConfigBase):
    def __init__(self):
        self.my_name = "RAZ"
        self.read_stdin_timeout = 0.1
        self.simulation_num_per_depth_about = 20
        self.hint_callback_per_sim = 10


class EvaluateConfig(ConfigBase):
    def __init__(self):
        self.game_num = 200  # 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 400
        self.play_config.thinking_loop = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.play_config.disable_resignation_rate = 0
        self.evaluate_latest_first = True


class PlayDataConfig(ConfigBase):
    def __init__(self):
        # Max Training Data Size = nb_game_in_file * max_file_num * 8
        self.multi_process_num = 16
        self.nb_game_in_file = 2
        self.max_file_num = 800
        self.save_policy_of_tau_1 = True
        self.enable_ggf_data = True
        self.nb_game_in_ggf_file = 100
        self.drop_draw_game_rate = 0


class PlayConfig(ConfigBase):
    def __init__(self):
        self.simulation_num_per_move = 200
        self.share_mtcs_info_in_self_play = True
        self.reset_mtcs_info_per_game = 1
        self.thinking_loop = 10
        self.required_visit_to_decide_action = 400
        self.start_rethinking_turn = 8
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.change_tau_turn = 4
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 8
        self.prediction_worker_sleep_sec  = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = -0.9
        self.allowed_resign_turn = 20
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01
        self.policy_decay_turn = 60  # not used
        self.policy_decay_power = 3

        # Using a solver is a kind of cheating!
        self.use_solver_turn = 50
        self.use_solver_turn_in_simulation = 50

        #
        self.schedule_of_simulation_num_per_move = [
            (0, 8),
            (300, 50),
            (2000, 200),
        ]

        # True means evaluating 'AlphaZero' method (disable 'eval' worker).
        # Please change to False if you want to evaluate 'AlphaGo Zero' method.
        self.use_newest_next_generation_model = True


class TrainerConfig(ConfigBase):
    def __init__(self):
        self.wait_after_save_model_ratio = 1  # wait after saving model
        self.batch_size = 256  # 2048
        self.min_data_size_to_learn = 100000
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 200
        self.use_tensorboard = True
        self.logging_per_steps = 100
        self.delete_self_play_after_number_of_training = 0  # control ratio of train:self data.
        self.lr_schedules = [
            (0, 0.01),
            (150000, 0.001),
            (300000, 0.0001),
        ]


class ModelConfig(ConfigBase):
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 10
        self.l2_reg = 1e-4
        self.value_fc_size = 256
