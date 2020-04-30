import numpy as np
import tensorflow as tf
from tqdm import tqdm

from minmax import OptimalActor
from mcts import MCTSActor
from tictactoe import TicTacToe
from game_base import GameBase, run_game
from neural_nets import ValueNetwork


if __name__ == "__main__":
    value_network = ValueNetwork()
    ttt_mcts = TicTacToe()
    mcts_actor = MCTSActor(ttt_mcts, value_network=value_network, num_expansions=300)

    optimal_data = np.load("data/minmax_cache.npy")
    optimal_actor = OptimalActor(optimal_data)


    ttt = TicTacToe()

    for i in range(10):
        if i % 2 == 0:
            data_dict = run_game(ttt, optimal_actor, mcts_actor)
        else:
            data_dict = run_game(ttt, mcts_actor, optimal_actor)

        result = data_dict["result"]
        if result == GameBase.Status.PLAYER1_WIN:
            value = 1
        elif result == GameBase.Status.PLAYER2_WIN:
            value = -1
        else:
            value = 0

        states = np.array(data_dict["states"], dtype=np.float32).reshape((-1, 3, 3, 1))
        
        


    


