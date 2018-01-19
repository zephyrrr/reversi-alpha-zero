class EvaluateConfig:
    def __init__(self):
        self.game_num = 50  # 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 1
        self.play_config.thinking_loop = 2
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.play_config.disable_resignation_rate = 0
        self.evaluate_latest_first = True


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 2000  # 50000
        self.save_policy_of_tau_1 = True


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 1
        self.thinking_loop = 2
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.change_tau_turn = 0
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 1
        self.prediction_worker_sleep_sec = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = -0.9
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01


class TrainerConfig:
    def __init__(self):
        self.batch_size = 512  # 2048
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 2000
        self.load_data_steps = 2000


class ModelConfig:
    cnn_filter_num = 32
    cnn_filter_size = 1
    res_layer_num = 1
    l2_reg = 1e-4
    value_fc_size = 32
