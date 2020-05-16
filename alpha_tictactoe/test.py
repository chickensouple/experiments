import tensorflow as tf
import numpy as np
from tqdm import tqdm

from minmax import OptimalActor
from mcts import MCTSActor
from tictactoe import TicTacToe
from game_base import GameBase, run_game
from neural_nets import ValueNetwork


# gpu setup copied from https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


if __name__ == "__main__":
    num_runs = 100
    num_expansions = 100

    ttt = TicTacToe()


    value_network = ValueNetwork()
    value_network.build((1, 3, 3, 1))
    value_network.load_weights("models/model.ckpt")

    ttt_mcts = TicTacToe()
    # mcts_actor = MCTSActor(ttt_mcts, num_expansions=num_expansions, value_network=value_network)
    mcts_actor = MCTSActor(ttt_mcts, num_expansions=num_expansions)

    optimal_data = np.load("data/minmax_cache.npy")
    optimal_actor = OptimalActor(optimal_data)


    data_dict = dict()
    data_dict[GameBase.Player.PLAYER1] = []
    data_dict[GameBase.Player.PLAYER2] = []
    for i in tqdm(range(num_runs)):
        if i % 2 == 0:
            mcts_player = GameBase.Player.PLAYER1
            actor1 = mcts_actor
            actor2 = optimal_actor
        else:
            mcts_player = GameBase.Player.PLAYER2
            actor1 = optimal_actor
            actor2 = mcts_actor
        result = run_game(ttt, actor1, actor2)

        data_dict[mcts_player].append(result["game_status"])


    data_dict[GameBase.Player.PLAYER1] = np.array(data_dict[GameBase.Player.PLAYER1])
    player1_wins = np.sum(data_dict[GameBase.Player.PLAYER1] == GameBase.Status.PLAYER1_WIN)
    player1_losses = np.sum(data_dict[GameBase.Player.PLAYER1] == GameBase.Status.PLAYER2_WIN)
    player1_ties = np.sum(data_dict[GameBase.Player.PLAYER1] == GameBase.Status.TIE)

    data_dict[GameBase.Player.PLAYER2] = np.array(data_dict[GameBase.Player.PLAYER2])
    player2_wins = np.sum(data_dict[GameBase.Player.PLAYER2] == GameBase.Status.PLAYER2_WIN)
    player2_losses = np.sum(data_dict[GameBase.Player.PLAYER2] == GameBase.Status.PLAYER1_WIN)
    player2_ties = np.sum(data_dict[GameBase.Player.PLAYER2] == GameBase.Status.TIE)

    all_wins = player1_wins + player2_wins
    all_losses = player1_losses + player2_losses
    all_ties = player1_ties + player2_ties

    print("Total Statistics\n============")
    print("wins: {}, losses: {}, ties: {}".format(all_wins, all_losses, all_ties))
    print("wins %: {}, losses %: {}, ties %: {}".format(
        all_wins / num_runs, 
        all_losses / num_runs, 
        all_ties / num_runs))

    print("\nPlayer 1 Statistics\n============")
    print("wins: {}, losses: {}, ties: {}".format(player1_wins, player1_losses, player1_ties))
    print("wins %: {}, losses %: {}, ties %: {}".format(
        player1_wins / len(data_dict[GameBase.Player.PLAYER1] * 100), 
        player1_losses / len(data_dict[GameBase.Player.PLAYER1] * 100), 
        player1_ties / len(data_dict[GameBase.Player.PLAYER1] * 100)))

    print("\nPlayer 2 Statistics\n============")
    print("wins: {}, losses: {}, ties: {}".format(player2_wins, player2_losses, player2_ties))
    print("wins %: {}, losses %: {}, ties %: {}".format(
        player2_wins / len(data_dict[GameBase.Player.PLAYER2] * 100), 
        player2_losses / len(data_dict[GameBase.Player.PLAYER2] * 100), 
        player2_ties / len(data_dict[GameBase.Player.PLAYER2] * 100)))

