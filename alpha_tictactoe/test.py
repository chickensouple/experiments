import numpy as np
from tqdm import tqdm

from minmax import OptimalActor
from mcts import MCTS
from tictactoe import TicTacToe
from game_base import GameBase

def run_game(mcts, minmax, game, mcts_player, num_expansions):
    game.reset()
    while game.get_game_status() == GameBase.Status.IN_PROGRESS:
        curr_player = game.get_curr_player()
        state = game.get_state()
        if (curr_player == mcts_player):
            action = mcts.search(state, mcts_player, num_expansions=num_expansions)
        else:
            action = minmax.get_action(state)
        game.step(action)

    outcome = game.get_outcome(mcts_player)
    return outcome

if __name__ == "__main__":
    num_runs = 100
    num_expansions = 1000

    ttt = TicTacToe()

    ttt_mcts = TicTacToe()
    mcts = MCTS(ttt_mcts)

    optimal_data = np.load("data/minmax_cache.npy")

    data_dict = dict()
    data_dict[GameBase.Player.PLAYER1] = []
    data_dict[GameBase.Player.PLAYER2] = []
    for i in tqdm(range(num_runs)):
        if i % 2 == 0:
            mcts_player = GameBase.Player.PLAYER1
            minmax_player = GameBase.Player.PLAYER2
        else:
            mcts_player = GameBase.Player.PLAYER2
            minmax_player = GameBase.Player.PLAYER1

        minmax = OptimalActor(minmax_player, optimal_data)

        result = run_game(mcts, minmax, ttt, mcts_player, num_expansions)
        data_dict[mcts_player].append(result)


    all_results = data_dict[GameBase.Player.PLAYER1] + data_dict[GameBase.Player.PLAYER2]
    all_results = np.array(all_results)
    all_wins = np.sum(all_results == 1)
    all_losses = np.sum(all_results == -1)
    all_ties = np.sum(all_results == 0)

    data_dict[GameBase.Player.PLAYER1] = np.array(data_dict[GameBase.Player.PLAYER1])
    player1_wins = np.sum(data_dict[GameBase.Player.PLAYER1] == 1)
    player1_losses = np.sum(data_dict[GameBase.Player.PLAYER1] == -1)
    player1_ties = np.sum(data_dict[GameBase.Player.PLAYER1] == 0)


    data_dict[GameBase.Player.PLAYER2] = np.array(data_dict[GameBase.Player.PLAYER2])
    player2_wins = np.sum(data_dict[GameBase.Player.PLAYER2] == 1)
    player2_losses = np.sum(data_dict[GameBase.Player.PLAYER2] == -1)
    player2_ties = np.sum(data_dict[GameBase.Player.PLAYER2] == 0)


    print("Total Statistics\n============")
    print("wins: {}, losses: {}, ties: {}".format(all_wins, all_losses, all_ties))
    print("wins %: {}, losses %: {}, ties %: {}".format(
        all_wins / len(all_results), 
        all_losses / len(all_results), 
        all_ties / len(all_results)))


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

