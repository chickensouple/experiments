import numpy as np
from tqdm import tqdm

from minmax import OptimalActor
from mcts import MCTSActor
from tictactoe import TicTacToe
from game_base import GameBase
from neural_nets import ValueNetwork


if __name__ == "__main__":
    value_network = ValueNetwork()
    ttt_mcts = TicTacToe()
    mcts_actor = MCTSActor(ttt_mcts, value_network=value_network, num_expansions=300)
    
    ttt = TicTacToe()
    mcts_actor.get_action(ttt.get_state())



