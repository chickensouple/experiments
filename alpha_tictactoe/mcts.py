import numpy as np
import copy
import math

from game_base import GameBase 
from tree import Tree

class NodeData(object):
    def __init__(self, 
        value_est=0, 
        num_visits=0, 
        game_state=None, 
        prev_action=None, 
        player=None):

        self.value_est = value_est
        self.num_visits = num_visits
        self.game_state = game_state
        self.prev_action = prev_action
        self.player = player
    
    def __str__(self):  
        return "(Player {}, Value: {}, Visits: {})".format(self.player, self.value_est, self.num_visits)

    def __repr__(self):
        return self.__str__()

class MCTS(object):
    def __init__(self, game):
        self.game = game
        self.tree = Tree()

    def compute_heuristic(self, node_data, player):
        value_est = node_data.value_est
        if (node_data.player != player):
            value_est = -value_est
        num_simulations = self.tree.get_node_data(0).num_visits
        heuristic = value_est + np.sqrt(num_simulations / (1 + node_data.num_visits))
        return heuristic

    def select(self, idx, player):
        children = self.tree.get_children(idx)
        if len(children) == 0:
            return None

        exploration_vals = []
        for child_idx in children:
            node_data = self.tree.get_node_data(child_idx)
            exp_val = self.compute_heuristic(node_data, player)
            exploration_vals.append(exp_val)  

        new_idx = children[np.argmax(exploration_vals)]
        return new_idx

    def simulate(self, game_state, player):
        self.game.set_state(game_state)
        while (self.game.get_game_status() == GameBase.Status.IN_PROGRESS):
            actions = self.game.get_valid_actions()
            rand_idx = np.random.randint(len(actions))
            action = actions[rand_idx]
            self.game.step(action)
        return self.game.get_outcome(player)

    def search(self, game_state, player, num_expansions=100):
        if self.tree.num_nodes() == 0:
            # add in root node
            initial_node = NodeData(
                value_est=0, 
                num_visits=0, 
                game_state=game_state, 
                prev_action=None,
                player=player)
            self.tree.insert_node(initial_node, None)
        else:
            # do breadth first search for a matching child
            def breadth_first_search(tree, game_state):
                child_list = []
                child_list += tree.get_children(0)
                
                found_idx = None
                child_list_idx = 0

                while child_list_idx < len(child_list):
                    child_idx = child_list[child_list_idx]
                    data = tree.get_node_data(child_idx)
                    if np.all(data.game_state == game_state):
                        found_idx = child_idx
                        break
                    child_list += tree.get_children(child_idx)
                    child_list_idx += 1
                return found_idx

            found_idx = breadth_first_search(self.tree, game_state)
            if found_idx is None:
                # add in root node
                initial_node = NodeData(
                    value_est=0, 
                    num_visits=0, 
                    game_state=game_state, 
                    prev_action=None,
                    player=player)
                self.tree.insert_node(initial_node, None)
            self.tree.rebase(found_idx)

        for _ in range(num_expansions):
            # select nodes in tree until leaf node
            curr_idx = 0
            idx_list = [curr_idx]
            curr_idx = self.select(curr_idx, player)
            while curr_idx is not None:
                idx_list.append(curr_idx)
                curr_idx = self.select(curr_idx, player)

            leaf_node_idx = idx_list[-1]
            
            # expand
            leaf_data = self.tree.get_node_data(leaf_node_idx)

            # if this node has never been expanded, add all next states
            # and then, choose one to be new leaf node to simulate
            if leaf_data.num_visits != 0:
                self.game.set_state(leaf_data.game_state)
                actions = self.game.get_valid_actions()
                if len(actions) != 0:
                    for action in actions:
                        self.game.set_state(leaf_data.game_state)
                        self.game.step(action)
                        new_node_data = NodeData(
                            value_est=0, 
                            num_visits=0, 
                            game_state=self.game.get_state(), 
                            prev_action=action,
                            player=self.game.get_curr_player())
                        self.tree.insert_node(new_node_data, leaf_node_idx)

                    leaf_node_idx = self.select(leaf_node_idx, player)
                    idx_list.append(leaf_node_idx)

            # simulate and update values for every node visited
            value_est = self.simulate(leaf_data.game_state, player)

            for idx in idx_list:
                node_data = self.tree.get_node_data(idx)
                node_data.value_est = float(value_est + node_data.num_visits * node_data.value_est) / (node_data.num_visits + 1)
                node_data.num_visits += 1
                self.tree.update_node_data(idx, node_data)
            
        children = self.tree.get_children(0)

        value_list = []
        for child_idx in children:
            child_data = self.tree.get_node_data(child_idx)
            value_list.append(child_data.value_est)
        best_idx = np.argmax(value_list)
        best_action = self.tree.get_node_data(children[best_idx]).prev_action 
        return best_action

if __name__ == "__main__":
    from tictactoe import TicTacToe
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimax TicTacToe Player. \
                     Use *generate* option to generate perform a search and cache the optimal actions.\
                     Then use the *play* option to read in the cached data and play a game against the computer.")
    parser.add_argument(
        "--player",
        action="store",
        type=int,
        default=1,
        choices=[1, 2],
        help="choose to play as player 1 or 2")
    args = parser.parse_args()

    if args.player == 1:
        computer_player = TicTacToe.GridState.PLAYER2
    else:
        computer_player = TicTacToe.GridState.PLAYER1

    ttt = TicTacToe()
    ttt_search = TicTacToe()
    mcts = MCTS(ttt_search)

    while ttt.get_game_status() == GameBase.Status.IN_PROGRESS:
        curr_player = ttt.get_curr_player()

        if curr_player == computer_player:
            
            state = ttt.get_state()
            action = mcts.search(state, computer_player, num_expansions=300)
            ttt.step(action)
            print(mcts.tree.get_node_data(0).num_visits)
        else:
            ttt.print_board()
            while True:
                action = input("Enter an action (row, col): ")
                try:
                    action = action.split(" ")
                    action = tuple([int(idx) for idx in action])

                    ttt.step(action)
                    break
                except Exception as e:
                    print(e)
                    print("Not an valid action. Try again.")

    ttt.print_board()
    print("End Game Status: {}".format(ttt.get_game_status().name))


