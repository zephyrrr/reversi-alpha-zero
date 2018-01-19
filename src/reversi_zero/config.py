import os


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir(config_type):
    return os.path.join(_project_dir(), "data_" + config_type)


class Config:
    def __init__(self, config_type="mini"):
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
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()

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


class Options:
    new = False


class ResourceConfig:
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

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard')
        self.force_learing_rate_file = os.path.join(self.data_dir, ".force-lr")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class GuiConfig:
    def __init__(self):
        self.window_size = (400, 440)
        self.window_title = "reversi-alpha-zero-generic"


class PlayWithHumanConfig:
    def __init__(self):
        self.simulation_num_per_move = 100
        self.thinking_loop = 1
        self.logging_thinking = True
        self.c_puct = 1
        self.parallel_search_num = 8
        self.noise_eps = 0
        self.change_tau_turn = 0
        self.resign_threshold = None
        self.use_newest_next_generation_model = True

    def update_play_config(self, pc):
        """

        :param reversi_zero.configs.normal.PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.thinking_loop = self.thinking_loop
        pc.logging_thinking = self.logging_thinking
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.change_tau_turn = self.change_tau_turn
        pc.parallel_search_num = self.parallel_search_num
        pc.resign_threshold = self.resign_threshold
        pc.use_newest_next_generation_model = self.use_newest_next_generation_model


class NBoardConfig:
    def __init__(self):
        self.my_name = "RAZ"
        self.read_stdin_timeout = 0.1
        self.simulation_num_per_depth_about = 20
        self.hint_callback_per_sim = 10
