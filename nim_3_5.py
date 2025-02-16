import datetime
import pathlib

import numpy as np
import torch

from .abstract_game import AbstractGame

import random

max_piles = 3
max_stones = 5

class MuZeroConfig:
    def __init__(self):
        # fmt: off

        self.seed = 0  # numpy、torch、ゲームのシード値
        self.max_num_gpus = None  # 使用する最大GPU数を設定。メモリが十分なら1つのGPU（1に設定）を使う方が通常は高速。Noneの場合、利用可能なすべてのGPUを使用

        ### ゲーム関連設定
        self.observation_shape = (2, max_piles, max_piles)  # Nimの観測（プレイヤー情報、山の状態）
        self.action_space = list(range(max_piles * max_stones + 1))  # 最大3つの山があるとして、山の番号と石の数を指定するリスト
        # [(i, j) for i in range(3) for j in range(1, 5)]
        self.players = list(range(2))  # 2人のプレイヤー
        self.stacked_observations = 0  # 現在の観測に追加する過去の観測や行動の数

        # 評価設定
        self.muzero_player = 0  # MuZeroが最初にプレイ
        self.opponent = "random"  # 対戦相手は「expert」

        ### セルフプレイ設定
        self.num_workers = 1  # リプレイバッファにデータを供給する同時実行スレッド数
        self.selfplay_on_gpu = False
        self.max_moves = max_piles * max_stones  # 手数の上限
        self.num_simulations = 50  # シミュレーション数
        self.discount = 1  # 報酬の時間割引率
        self.temperature_threshold = (max_piles * max_stones) // 3  # 手数がこの値を超えた後は、visit_softmax_temperature_fnによる温度が0になり、最善手を選択する。Noneの場合は常に関数を使用

        # ルート探索ノイズ
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB（Upper Confidence Bound）公式のパラメータ
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### ネットワーク設定
        self.network = "fullyconnected"  # "resnet" / "fullyconnected" のいずれか
        self.support_size = 10

        # 残差ネットワーク
        self.downsample = False  # 表現ネットワークの前で観測をダウンサンプリングするか。False / "CNN" / "resnet"
        self.blocks = 1  # ResNetのブロック数
        self.channels = 16  # ResNetのチャネル数
        self.reduced_channels_reward = 16  # 報酬ヘッドのチャネル数
        self.reduced_channels_value = 16  # 価値ヘッドのチャネル数
        self.reduced_channels_policy = 16  # ポリシーヘッドのチャネル数
        self.resnet_fc_reward_layers = [8]  # 動的ネットワークの報酬ヘッドの隠れ層
        self.resnet_fc_value_layers = [8]  # 予測ネットワークの価値ヘッドの隠れ層
        self.resnet_fc_policy_layers = [8]  # 予測ネットワークのポリシーヘッドの隠れ層

        # 全結合ネットワーク
        self.encoding_size = 16
        self.fc_representation_layers = [32, 32]  # 表現ネットワークの隠れ層
        self.fc_dynamics_layers = [16]  # 動的ネットワークの隠れ層
        self.fc_reward_layers = [8]  # 報酬ネットワークの隠れ層
        self.fc_value_layers = []  # 価値ネットワークの隠れ層
        self.fc_policy_layers = []  # ポリシーネットワークの隠れ層

        ### トレーニング設定
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 80000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.5
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"  # "Adam" または "SGD"。論文ではSGDを使用
        self.weight_decay = 1e-4  # L2正則化
        self.momentum = 0.9  # オプティマイザがSGDの場合に使用

        # 学習率の指数関数的スケジュール
        self.lr_init = 0.005  # 初期学習率
        self.lr_decay_rate = 1  # 定数学習率を使用する場合は1に設定
        self.lr_decay_steps = 10000  # 学習率を減衰させるステップ数

        ### リプレイバッファ
        self.replay_buffer_size = 1000  # リプレイバッファに保持するセルフプレイゲームの数
        self.num_unroll_steps = max_piles * max_stones  # 各バッチ要素で保持するゲームの手数
        self.td_steps = 10  # ターゲット値を計算するために考慮する未来のステップ数
        self.PER = True  # 優先度付きリプレイを使用
        self.PER_alpha = 0.8  # 優先度の度合い

        # 再解析設定
        self.use_last_model_value = True  # 最新モデルで安定したnステップ価値を提供
        self.reanalyse_on_gpu = False

        ### セルフプレイとトレーニングのバランス調整
        self.self_play_delay = 0  # 各セルフプレイ後の待機秒数
        self.training_delay = 0  # 各トレーニングステップ後の待機秒数
        self.ratio = 1  # トレーニングステップとセルフプレイステップの比率
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        訓練が進むにつれて、行動選択が貪欲になるように訪問回数の分布を調整するパラメータ。

        Returns:
            正の浮動小数点数。
        """
        return 1
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
        """
    


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = Nim()

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        return 0 if self.env.player == 1 else 1

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        input("Press enter to take a step")

    def human_to_action(self):
        while True:
            try:
                pile = int(input("Enter the pile number (1-3): "))-1
                num = int(input(f"Enter the number of stones to remove from pile {pile+1}: "))
                if (pile * max_stones + num) in self.legal_actions():
                    break
            except:
                pass
            print("Wrong input, try again")
        return (pile * max_stones + num)

    def expert_agent(self):
        return self.env.expert_action()

    def action_to_string(self, action):
        pile = (action - 1) // max_stones
        num = (action - 1) % max_stones + 1
        return f"Remove {num} stones from pile {pile+1}"



class Nim:
    def __init__(self):
        self.piles = [random.randint(1, max_stones) for _ in range(random.randint(max_piles, max_piles))]
        """
        a = 0
        b = 0
        while a == b:
            a = random.randint(1, max_stones)
            b = random.randint(1, max_stones)
        self.piles = [a, b, 0]
        """
        self.player = 1  # プレイヤー1は1、プレイヤー2は-1で示す
        self.re = 0

    def to_play(self):
        """
        現在のプレイヤーを返す。
        
        戻り値:
            プレイヤー1なら0、プレイヤー2なら1。
        """
        return 0 if self.player == 1 else 1

    def reset(self):
        """
        ゲームを初期化する。
        
        戻り値:
            初期化後のゲームの観測。
        """
        self.piles = [random.randint(1, max_stones) for _ in range(random.randint(max_piles, max_piles))]
        """
        a = 0
        b = 0
        while a == b:
            a = random.randint(1, max_stones)
            b = random.randint(1, max_stones)
        self.piles = [a, b, 0]
        """
        self.player = 1
        return self.get_observation()

    def step(self, action):
        """
        指定された行動を適用する。

        引数:
            action: (pile, num) の形式で、どの山から何個の石を取るか。

        戻り値:
            観測、報酬、およびゲームが終了したかどうかの真偽値。
        """
        pile = (action - 1) // max_stones
        num = (action - 1) % max_stones + 1
        self.piles[pile] -= num

        # ゲーム終了の判定
        done = all(pile == 0 for pile in self.piles)
        reward = 1 if done and self.player == 1 else 0 if done else 0

        # プレイヤーの交代
        self.player *= -1
        
        self.re = reward
        return self.get_observation(), reward, done

    def get_observation(self):
        """
        現在の山の状態を観測として取得する。

        戻り値:
            各山の石の数と次にプレイするプレイヤーの情報を含む配列。
        """
        piles_state = np.array([self.piles], dtype="float32")
        expanded_pile = np.tile(piles_state, (max_piles, 1))
        player_state = np.full((max_piles, max_piles), self.player, dtype="float32")
        return np.array([expanded_pile, player_state])
        

    def legal_actions(self):
        """
        合法な行動をリストとして返す。

        戻り値:
            合法な行動の (山のインデックス, 石の数) のタプルのリスト。
        """
        actions = []
        for pile_idx, stones in enumerate(self.piles):
            if stones > 0:
                for num in range(1, stones + 1):
                    actions.append(pile_idx * max_stones + num)
        return actions

    def have_winner(self):
        """
        勝者がいるかどうかを判定する。

        戻り値:
            ゲーム終了であればTrue、続行ならFalse。
        """
        return all(pile == 0 for pile in self.piles)

    def expert_action(self):
        """
        最適な行動を選択する。

        戻り値:
            現在の状態に基づいて選択された行動のタプル。
        """
        x = 0
        for pile in self.piles:
            x ^= pile

        # XORが0ならランダムな合法手を取る
        if x == 0:
            return random.choice(self.legal_actions())

        # XORが0でないなら、勝利するための最善手を探す
        for pile_idx, stones in enumerate(self.piles):
            target = stones ^ x
            if target < stones:
                num = stones - target
                return (pile_idx * max_stones + num)

    def render(self):
        """
        現在の山の状態を表示する。
        """
        print("Piles:", self.piles)
        #print(self.get_observation())
        print(self.legal_actions())
        x = 0
        for i in range(len(self.piles)):
            x ^= self.piles[i]
        print(f"XOR={x}, next_player = {self.player}, reward = {self.re}")
        print()
