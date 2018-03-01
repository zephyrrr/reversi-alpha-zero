from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np
from numpy.random import random

from reversi_zero.agent.api import ReversiModelAPI
from reversi_zero.config import Config

from reversi_zero.env.reversi_env import Board
from reversi_zero.env.reversi_env import ReversiEnv, Player, Winner, another_player
#from reversi_zero.lib.bitboard import bit_to_array, flip_vertical, rotate90
from reversi_zero.lib.bitboard import dirichlet_noise_of_mask
# from reversi_zero.lib.reversi_solver import ReversiSolver
from reversi_zero.lib.alt.reversi_solver import ReversiSolver

#import uvloop
#asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
#from profilehooks import profile


CounterKey = namedtuple("CounterKey", "board_data next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit enemy_values enemy_visit")
CallbackInMCTS = namedtuple("CallbackInMCTS", "per_sim callback")
MCTSInfo = namedtuple("MCTSInfo", "var_n var_w var_p")
ActionWithEvaluation = namedtuple("ActionWithEvaluation", "action n q")

logger = getLogger(__name__)


class ReversiPlayer:
    def __init__(self, config: Config, model, play_config=None, enable_resign=True, mtcs_info=None, api=None):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel|None model:
        :param MCTSInfo mtcs_info:
        :parameter ReversiModelAPI api:
        """
        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.enable_resign = enable_resign
        self.use_pure_mcts = False
        self.api = api or ReversiModelAPI(self.config, self.model)

        # key=(board_data, action)
        mtcs_info = mtcs_info or self.create_mtcs_info()
        self.var_n, self.var_w, self.var_p = mtcs_info

        self.expanded = set(self.var_p.keys())
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.callback_in_mtcs = None

        self.thinking_history = {}  # for fun
        self.resigned = False
        self.requested_stop_thinking = False
        self.solver = self.create_solver()

    @staticmethod
    def rollout_policy_fn(legal_moves):
        """rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        legal_moves_arr = Board.to_n_labels(legal_moves)
        action_probs = np.random.rand(Board.n_labels) * legal_moves_arr
        return action_probs

    @staticmethod
    def policy_value_fn(legal_moves):
        """a function that takes in a state and outputs a list of (action, probability)
        tuples and a score for the state"""
        # return uniform probabilities and 0 score for pure MCTS
        legal_moves_arr = Board.to_n_labels(legal_moves)

        legal_moves_list = [item for sublist in legal_moves for item in sublist]
        len_legal = legal_moves_list.count(1)
        action_probs = legal_moves_arr / len_legal

        return action_probs

    def _evaluate_rollout(self, env, limit=1000):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        player = env.next_player
        for i in range(limit):
            if env.done:
                break
            action_probs = self.rollout_policy_fn(env.legal_moves())
            action_t = int(np.argmax(action_probs))
            env.step(action_t)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if not env.done:
            return 0
        if env.winner == Winner.draw:  # tie
            return 0
        elif env.winner == Winner.black and player == Player.black:
            return 1
        elif env.winner == Winner.white and player == Player.white:
            return 1
        else:
            return -1

    @staticmethod
    def create_mtcs_info():
        return MCTSInfo(defaultdict(lambda: np.zeros((Board.n_labels,))),
                        defaultdict(lambda: np.zeros((Board.n_labels,))),
                        defaultdict(lambda: np.zeros((Board.n_labels,))))

    def var_q(self, key):
        return self.var_w[key] / (self.var_n[key] + 1e-5)

    def reset(self, enable_resign=True):
        self.moves.clear()
        self.thinking_history.clear()
        self.enable_resign = enable_resign

    def action(self, board_data, callback_in_mtcs=None):
        """

        :param own: BitBoard
        :param enemy:  BitBoard
        :param CallbackInMCTS callback_in_mtcs:
        :return action=move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right)
        """
        action_with_eval = self.action_with_evaluation(own, enemy, callback_in_mtcs=callback_in_mtcs)
        return action_with_eval.action

    def action_with_evaluation(self, board_data, callback_in_mtcs=None):
        """

        :param own: BitBoard
        :param enemy:  BitBoard
        :param CallbackInMCTS callback_in_mtcs:
        :rtype: ActionWithEvaluation
        :return ActionWithEvaluation(
                    action=move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right),
                    n=N of the action,
                    q=W/N of the action,
                )
        """
        env = ReversiEnv().update(board_data, Player.black)
        key = self.counter_key(env)
        self.callback_in_mtcs = callback_in_mtcs
        pc = self.play_config

        if pc.use_solver_turn and env.turn >= pc.use_solver_turn:
            ret = self.action_by_searching(key)
            if ret:  # not save move as play data
                return ret

        for tl in range(self.play_config.thinking_loop):
            if env.turn > 0:
                self.search_moves(own, enemy)
            else:
                self.bypass_first_move(key)

            policy = self.calc_policy(own, enemy)
            action = int(np.random.choice(range(64), p=policy))
            action_by_value = int(np.argmax(self.var_q(key) + (self.var_n[key] > 0)*100))
            value_diff = self.var_q(key)[action] - self.var_q(key)[action_by_value]

            if env.turn <= pc.start_rethinking_turn or self.requested_stop_thinking or \
                    (value_diff > -0.01 and self.var_n[key][action] >= pc.required_visit_to_decide_action):
                break

        # this is for play_gui, not necessary when training.
        self.update_thinking_history(env, board_data, action, policy)

        if self.play_config.resign_threshold is not None and\
                        np.max(self.var_q(key) - (self.var_n[key] == 0)*10) <= self.play_config.resign_threshold:
            self.resigned = True
            if self.enable_resign:
                if env.turn >= self.config.play.allowed_resign_turn:
                    return ActionWithEvaluation(None, 0, 0)  # means resign
                else:
                    logger.debug(f"Want to resign but disallowed turn {env.turn} < {self.config.play.allowed_resign_turn}")

        saved_policy = self.calc_policy_by_tau_1(key) if self.config.play_data.save_policy_of_tau_1 else policy
        self.add_data_to_move_buffer_with_8_symmetries(env, saved_policy)
        return ActionWithEvaluation(action=action, n=self.var_n[key][action], q=self.var_q(key)[action])

    def update_thinking_history(self, env, board_data, action, policy):
        key = CounterKey(board_data, Player.black.value)
        next_key = self.get_next_key(board_data, action)
        self.thinking_history[env.get_hashable_board_data()] = \
            HistoryItem(action, policy, list(self.var_q(key)), list(self.var_n[key]),
                        list(self.var_q(next_key)), list(self.var_n[next_key]))

    def bypass_first_move(self, key):
        legal_array = bit_to_array(find_correct_moves(key.black, key.white), 64)
        action = np.argmax(legal_array)
        self.var_n[key][action] = 1
        self.var_w[key][action] = 0
        self.var_p[key] = legal_array / np.sum(legal_array)

    def action_by_searching(self, key):
        action, score = self.solver.solve(key.black, key.white, Player(key.next_player), exactly=True)
        if action is None:
            return None
        # logger.debug(f"action_by_searching: score={score}")
        policy = np.zeros(64)
        policy[action] = 1
        self.var_n[key][action] = 999
        self.var_w[key][action] = np.sign(score) * 999
        self.var_p[key] = policy
        self.update_thinking_history(key.black, key.white, action, policy)
        return ActionWithEvaluation(action=action, n=999, q=np.sign(score))

    def stop_thinking(self):
        self.requested_stop_thinking = True

    def add_data_to_move_buffer_with_8_symmetries(self, env, policy):
        for flip in [False, True]:
            if flip and not Board.is_symmetry:
                continue
            for rot_right in range(4):
                if rot_right and not Board.is_symmetry:
                    continue
                env_saved = ReversiEnv().update(env.get_board_data(), env.next_player)
                policy_saved = policy.reshape((Board.height, Board.width))

                if flip:
                    env_saved = env_saved.flip_vertical()
                    policy_saved = np.flipud(policy_saved)
                if rot_right:
                    for _ in range(rot_right):
                        env_saved = env_saved.rotate90()
                    policy_saved = np.rot90(policy_saved, k=-rot_right)
                self.moves.append([env_saved.get_hashable_board_data(), list(policy_saved.reshape((Board.n_labels, )))])

    def get_next_key(self, board_data, action):
        env = ReversiEnv().update(board_data, Player.black)
        env.step(action)
        return self.counter_key(env)

    def ask_thought_about(self, board_data) -> HistoryItem:
        env = ReversiEnv().update(board_data, Player.black)
        return self.thinking_history.get(env.get_hashable_board_data())

    #@profile
    def search_moves(self, board_data):
        loop = self.loop
        self.running_simulation_num = 0
        self.requested_stop_thinking = False

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(board_data)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, board_data):
        self.running_simulation_num += 1
        root_key = self.counter_key(ReversiEnv().update(board_data, Player.black))
        with await self.sem:  # reduce parallel search number
            if self.requested_stop_thinking:
                self.running_simulation_num -= 1
                return None
            env = ReversiEnv().update(board_data, Player.black)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            if self.callback_in_mtcs and self.callback_in_mtcs.per_sim > 0 and \
                    self.running_simulation_num % self.callback_in_mtcs.per_sim == 0:
                self.callback_in_mtcs.callback(list(self.var_q(root_key)), list(self.var_n[root_key]))
            return leaf_v

    async def search_my_move(self, env: ReversiEnv, is_root_node=False):
        """
        Q, V is value for this Player(always black).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return:
        """

        if env.done:
            if env.winner == Winner.black:
                return 1
            elif env.winner == Winner.white:
                return -1
            elif env.winner == Winner.draw:
                return 0
            else:
                raise Exception("not supported winner")

        key = self.counter_key(env)
        another_side_key = self.another_side_counter_key(env)

        if self.config.play.use_solver_turn_in_simulation and \
                env.turn >= self.config.play.use_solver_turn_in_simulation:
            action, score = self.solver.solve(key.black, key.white, Player(key.next_player), exactly=False)
            if action:
                score = score if env.next_player == Player.black else -score
                leaf_v = np.sign(score)
                leaf_p = np.zeros(64)
                leaf_p[action] = 1
                self.var_n[key][action] += 1
                self.var_w[key][action] += leaf_v
                self.var_p[key] = leaf_p
                self.var_n[another_side_key][action] += 1
                self.var_w[another_side_key][action] -= leaf_v
                self.var_p[another_side_key] = leaf_p
                return np.sign(score)

        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            if self.use_pure_mcts:
                leaf_v = await self.expand_and_evaluate_pure(env)
            else:
                leaf_v = await self.expand_and_evaluate(env)
            if env.next_player == Player.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black

        virtual_loss = self.config.play.virtual_loss
        #virtual_loss = 0
        virtual_loss_for_w = virtual_loss if env.next_player == Player.black else -virtual_loss

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss_for_w
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path
        # update: N, W
        self.var_n[key][action_t] += - virtual_loss + 1
        self.var_w[key][action_t] += virtual_loss_for_w + leaf_v
        # update another side info(flip color and player)
        self.var_n[another_side_key][action_t] += 1
        self.var_w[another_side_key][action_t] -= leaf_v  # must flip the sign.
        return leaf_v

    async def expand_and_evaluate_pure(self, env):
        key = self.counter_key(env)
        self.now_expanding.add(key)

        env_saved = ReversiEnv().update(env.get_board_data(), env.next_player)
        leaf_v = self._evaluate_rollout(env_saved)

        leaf_p = self.policy_value_fn(env.legal_moves())
        self.var_p[key] = leaf_p
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    #@profile
    async def expand_and_evaluate(self, env):
        """expand new leaf
        update var_p, return leaf_v
        :param ReversiEnv env:
        :return: leaf_v
        """
        key = self.counter_key(env)
        another_side_key = self.another_side_counter_key(env)
        self.now_expanding.add(key)

        if Board.is_symmetry:
            # (di(p), v) = fθ(di(sL))
            # rotation and flip. flip -> rot.
            is_flip_vertical = random() < 0.5
            rotate_right_num = int(random() * 4)
        else:
            is_flip_vertical = False
            rotate_right_num = 0

        #is_flip_vertical = True
        #rotate_right_num = 1

        # black, white = env.black_and_white_plane()  # env.board.black, env.board.white
        # if is_flip_vertical:
        #     black, white = Board.flip_vertical(black), Board.flip_vertical(white)
        # for i in range(rotate_right_num):
        #     black, white = Board.rotate90(black), Board.rotate90(white)  # rotate90: rotate bitboard RIGHT 1 time
        # black_ary, white_ary = np.array(black), np.array(white)

        env_saved = ReversiEnv().update(env.get_board_data(), env.next_player)
        if is_flip_vertical:
            env_saved = env_saved.flip_vertical()
        for i in range(rotate_right_num):
            env_saved = env_saved.rotate90()
        black, white = env_saved.get_board_data()
        black_ary, white_ary = np.array(black), np.array(white)

        state = [black_ary, white_ary] if env.next_player == Player.black else [white_ary, black_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        # reverse rotate and flip about leaf_p
        if rotate_right_num > 0 or is_flip_vertical:  # reverse rotation and flip. rot -> flip.
            leaf_p = leaf_p.reshape((Board.height, Board.width))
            if rotate_right_num > 0:
                leaf_p = np.rot90(leaf_p, k=rotate_right_num)  # rot90: rotate matrix LEFT k times
            if is_flip_vertical:
                leaf_p = np.flipud(leaf_p)
            leaf_p = leaf_p.reshape((Board.n_inputs, ))

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.var_p[another_side_key] = leaf_p
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            #logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)  # shape=(N, 2, 8, 8)
            #logger.debug(f"predicted {len(item_list)} items")
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        """
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def calc_policy(self, board_data):
        """calc π(a|s0)
        :param own:
        :param enemy:
        :return:
        """
        pc = self.play_config
        env = ReversiEnv().update(board_data, Player.black)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.calc_policy_by_tau_1(key)
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(Board.n_labels)
            ret[action] = 1
            return ret

    def calc_policy_by_tau_1(self, key):
        s = np.sum(self.var_n[key])
        if s == 0:
            return self.var_n[key]
        return self.var_n[key] / s  # tau = 1

    @staticmethod
    def counter_key(env: ReversiEnv):
        return CounterKey(env.get_hashable_board_data(), env.next_player.value)

    @staticmethod
    def another_side_counter_key(env: ReversiEnv):
        return CounterKey(env.board.white, env.board.black, another_player(env.next_player).value)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)

        legal_moves = env.legal_moves()
        legal_moves_arr = Board.to_n_labels(legal_moves)

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0

        p_ = self.var_p[key]

        # re-normalize in legal moves
        p_ = p_ * legal_moves_arr
        if np.sum(p_) > 0:
            # decay policy gradually in the end phase
            _pc = self.config.play
            temperature = min(np.exp(1-np.power(env.turn/_pc.policy_decay_turn, _pc.policy_decay_power)), 1)
            # normalize and decay policy
            p_ = self.normalize(p_, temperature)

        if is_root_node and self.play_config.noise_eps > 0:  # Is it correct?? -> (1-e)p + e*Dir(alpha)
            noise = dirichlet_noise_of_mask(legal_moves, self.play_config.dirichlet_alpha)
            p_ = (1 - self.play_config.noise_eps) * p_ + self.play_config.noise_eps * noise

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.next_player == Player.black:
            v_ = (self.var_q(key) + u_ + 1000) * legal_moves_arr
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q(key) + u_ + 1000) * legal_moves_arr

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

    @staticmethod
    def normalize(p, t=1):
        pp = np.power(p, t)
        return pp / np.sum(pp)

    def create_solver(self):
        return ReversiSolver()

