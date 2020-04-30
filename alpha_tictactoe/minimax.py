import numpy as np
import copy
import time

from tree import Tree
from game_base import GameBase

class MiniMaxNodeState(object):
    def __init__(self,
        game_state,
        minimax_value,
        optimal_action):
        self.game_state = game_state
        self.minimax_value = minimax_value
        self.optimal_action = optimal_action

class MinMaxSearchTree(object):
    """
    MinMax Search Tree
    """
    def __init__(self, game):
        """
        ctor for min max search tree.

        Arguments:
            game {GameBase} -- A game object that will be used and mutated by the minmax tree search.
                Don't the game outside of this class as it will be manipulated within here.
        """
        self.tree = Tree()
        self.game = game

    def search(self, game_state, player):
        """
        Perform Minimax search and return the search tree.
        The returned search tree will contain a tuple of data at each node.
        This tuple consists of (game_state, minimax_value, optimal_action)

        Arguments:
            game_state {np.array} -- state of the game as returned by ttt.get_state()
            player {which player to solve minimax tree for} -- PLAYER1 or PLAYER2
        """
        # clear out any previous searches
        self.tree.reset()

        # Insert the parent node
        root_idx = self.tree.insert_node(MiniMaxNodeState(game_state, None, None), None)

        # Start expanding from parent node
        self._expand_node(root_idx, player)
        return self.tree

    def _expand_node(self, node_idx, player):
        # get possible actions
        node_data = self.tree.get_node_data(node_idx)
        self.game.set_state(node_data.game_state)
        curr_player = self.game.get_curr_player()
        actions = self.game.get_valid_actions()

        # If we have reached a leaf node, get the value and return
        # 1 for winning, -1 for losing, 0 for tie
        if len(actions) == 0:
            val = self.game.get_outcome(player)
            node_data.minimax_value = val
            self.tree.update_node_data(node_idx, node_data)
            return val

        # Recursively expand each child node
        # and collect the minimax values
        minimax_vals = []
        for action in actions:
            self.game.set_state(node_data.game_state)
            self.game.step(action)

            new_node_idx = self.tree.insert_node(MiniMaxNodeState(self.game.get_state(), None, None), node_idx)
            val = self._expand_node(new_node_idx, player)
            minimax_vals.append(val)

        # Compute minimum or maximum of values depending on what level the
        # search is currently on
        if player == curr_player:
            val_idx = np.argmax(minimax_vals)
        else:
            val_idx = np.argmin(minimax_vals)
        val = minimax_vals[val_idx]
        opt_action = actions[val_idx]

        # update the expanded node with the value and optimal action
        node_data.minimax_value = val
        node_data.optimal_action = opt_action
        self.tree.update_node_data(node_idx, node_data)
        return val


class TicTacToeMinMaxPlayer(object):
    """
    Class to play tic tac toe using MinMax search results.
    """
    def __init__(self, player, optimal_data):
        """
        Creates the min max player.

        Arguments:
            player {GameBase.Player} -- Enum for which player this will be
            optimal_data {np.array} -- The numpy array generated by TicTacToeMinMaxPlayer.generate_optimal_data() 
        """
        self.player = player
        self.optimal_data = optimal_data

    def get_action(self, game_state):
        """
        Get an action at a particular game state.
        """
        idx = TicTacToeMinMaxPlayer._state_to_idx(game_state)
        return tuple(self.optimal_data[idx, :])

    _POWERS_3 = np.power(3, np.arange(9))
    _PLAYER_OFFSET = np.sum(2 * _POWERS_3) + 1
    _MAX_STATES = _PLAYER_OFFSET + np.sum(2 * _POWERS_3)
    @staticmethod
    def _state_to_idx(state):
        idx = np.sum(TicTacToeMinMaxPlayer._POWERS_3 * state[:9])
        idx += (state[9] == GameBase.Player.PLAYER2) * TicTacToeMinMaxPlayer._PLAYER_OFFSET
        return idx

    @staticmethod
    def generate_optimal_data():
        """
        Generates numpy array of optimal moves.
        It will be an (N, 2) array. Where the i'th row is the optimal action 
        for the i'th state. The states are indexed by flattening the state using
        _state_to_idx().
        """
        ttt = TicTacToe()
        ttt_search = TicTacToe()
        search_tree = MinMaxSearchTree(ttt_search)

        # Run search for the various scenarios where the minmax player has to go first
        # or second reacting to various first moves.
        tree_list = []
        tree_list.append(copy.deepcopy(search_tree.search(ttt.get_state(), GameBase.Player.PLAYER1)))
        actions = ttt.get_valid_actions()
        initial_state = ttt.get_state()
        for action in actions:
            ttt.set_state(initial_state)
            ttt.step(action)
            tree_list.append(copy.deepcopy(search_tree.search(ttt.get_state(), GameBase.Player.PLAYER2)))

        # Take the search trees and condense the optimal actions into a numpy array
        optimal_actions = np.ones((TicTacToeMinMaxPlayer._MAX_STATES, 2), dtype=np.int8) * -1
        for tree in tree_list:
            for node in tree.nodes:
                idx = TicTacToeMinMaxPlayer._state_to_idx(node.game_state)
                if node.optimal_action != None:
                    optimal_actions[idx, :] = node.optimal_action
        return optimal_actions

if __name__ == "__main__":
    import argparse
    import pickle
    from tictactoe import TicTacToe

    parser = argparse.ArgumentParser(
        description="Minimax TicTacToe Player. \
                     Use *generate* option to generate perform a search and cache the optimal actions.\
                     Then use the *play* option to read in the cached data and play a game against the computer.")
    parser.add_argument(
        "--file",
        action="store",
        type=str,
        default="/tmp/minmax_cache.npy",
        help="File to store/load search trees.")
    subparser = parser.add_subparsers(
        help="Generate tree or play game.",
        dest="cmd")
    generate_subparser = subparser.add_parser("generate",
        help="Generate Search Trees and save them.")

    play_subparser = subparser.add_parser("play",
        help="Play against minimax computer.")
    play_subparser.add_argument(
        "--player",
        action="store",
        type=int,
        default=1,
        choices=[1, 2],
        help="choose to play as player 1 or 2")
    args = parser.parse_args()

    if args.cmd == "generate":
        start_time = time.clock()
        optimal_data = TicTacToeMinMaxPlayer.generate_optimal_data()
        end_time = time.clock()
        print("Total time for full minimax search: {} seconds".format(end_time - start_time))

        np.save(args.file, optimal_data)
    else:
        optimal_data = np.load(args.file)

        if args.player == 1:
            computer_player = GameBase.Player.PLAYER2
        else:
            computer_player = GameBase.Player.PLAYER1
        minimax_player = TicTacToeMinMaxPlayer(computer_player, optimal_data)

        ttt = TicTacToe()
        print("Actions are input as a 2 numbers seperated by a space, indicating the row and column.")
        while ttt.get_game_status() == GameBase.Status.IN_PROGRESS:
            curr_player = ttt.get_curr_player()
            if curr_player == computer_player:
                action = minimax_player.get_action(ttt.get_state())
                ttt.step(action)
            else:
                ttt.print_board()
                action = input("Enter an action (row, col): ")
                try:
                    action = action.split(" ")
                    action = tuple([int(idx) for idx in action])
                    ttt.step(action)
                except Exception as e:
                    print(e)
                    print("Not an valid action. Try again.")
                    print("Actions are input as a 2 numbers seperated by a space, indicating the row and column.")
                    continue
        ttt.print_board()
        print("End Game Status: {}".format(ttt.get_game_status().name))
