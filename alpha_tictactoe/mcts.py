import numpy as np
import copy
import math

from game_base import GameBase, GameActor
from tree import Tree

class MCTSNodeData(object):
    """
    Data contained in a node of the MCTS
    Attributes:
        value_est {float} -- Takes on values [-1, 1]
            This is the estimate of the value of the game_state at this node.
            1 means it is estimated player 1 will win with 100% probability.
            0 means it is estimated that it will be a tie with 100% probability.
            -1 means it is estimated player2 will win with 100% probability.
        num_visits {int} -- The number of times this node has been updated.
        game_state -- the state of the game given by GameBase.get_state()
        prev_action -- the action taken from parent node to arrive at this node.
        player {GameBase.Player} -- which player's turn is at this node.
        value_net_cache {float} -- cached value of the value_network
    """
    def __init__(self, 
        value_est=0, 
        num_visits=0, 
        game_state=None, 
        prev_action=None, 
        player=None,
        value_net_cache=None):

        self.value_est = value_est
        self.num_visits = num_visits
        self.game_state = game_state
        self.prev_action = prev_action
        self.player = player
        self.value_net_cache = value_net_cache
    
    def __str__(self):  
        return "(Player {}, Value: {}, Visits: {}, Prev Action: {})".format(
            self.player, 
            self.value_est, 
            self.num_visits,
            self.prev_action)

    def __repr__(self):
        return self.__str__()

class MCTSActor(GameActor):
    def __init__(self, game, num_expansions=300, value_network=None):
        self.game = game
        self.num_expansions = num_expansions
        self.value_network = value_network

        self.tree = Tree()

    def compute_heuristic(self, node_data):
        value_est = node_data.value_est
        if (node_data.player == GameBase.Player.PLAYER2):
            value_est = -value_est
        num_simulations = self.tree.get_node_data(0).num_visits
        heuristic = value_est + np.sqrt(num_simulations / (1 + node_data.num_visits))
        return heuristic

    def select(self, idx):
        children = self.tree.get_children(idx)
        if len(children) == 0:
            return None

        exploration_vals = []
        for child_idx in children:
            node_data = self.tree.get_node_data(child_idx)
            exp_val = self.compute_heuristic(node_data)
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

        status = self.game.get_game_status()
        if status == GameBase.Status.PLAYER1_WIN:
            return 1
        elif status == GameBase.Status.PLAYER2_WIN:
            return -1
        else:
            return 0


    def _check_tree(self, idx, level=0):
        data = self.tree.get_node_data(idx)
        self.game.set_state(data.game_state)
        children = self.tree.get_children(idx)
        actions = self.game.get_valid_actions()
        
        valid_list = []
        for child_idx in children:
            prev_action = self.tree.get_node_data(child_idx).prev_action
            if prev_action not in actions:
                print("========\nFailure:\n{}\nlevel: {}\nprev_action: {}\nidx: {}\n".format(
                    data.game_state[:9].reshape((3, 3)),
                    level,
                    prev_action,
                    child_idx
                ))
                return False
            valid_list.append(self._check_tree(child_idx, level+1))

        valid = np.all(valid_list)
        return valid

    def get_action(self, game_state):
        self.game.set_state(game_state)
        curr_player = self.game.get_curr_player()



        if self.tree.num_nodes() == 0:
            # add in root node
            initial_node = MCTSNodeData(
                value_est=0, 
                num_visits=0, 
                game_state=game_state, 
                prev_action=None,
                player=curr_player)
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
                # clear tree
                self.tree.reset()

                # add in root node
                initial_node = MCTSNodeData(
                    value_est=0, 
                    num_visits=0, 
                    game_state=game_state, 
                    prev_action=None,
                    player=curr_player)
                self.tree.insert_node(initial_node, None)
            else:
                # prev_tree = copy.deepcopy(self.tree)
                self.tree.rebase(found_idx)


        for _ in range(self.num_expansions):
            # select nodes in tree until leaf node
            curr_idx = 0
            idx_list = [curr_idx]
            curr_idx = self.select(curr_idx)
            while curr_idx is not None:
                idx_list.append(curr_idx)
                curr_idx = self.select(curr_idx)

            leaf_node_idx = idx_list[-1]
            
            # expand
            leaf_data = self.tree.get_node_data(leaf_node_idx)

            # this node has never been visited yet, don't expand, just simulate().
            # If it has been visited and is a leaf node, expand the node
            # and choose a new leaf node from the children actions to simulate.
            if leaf_data.num_visits != 0:
                self.game.set_state(leaf_data.game_state)
                actions = self.game.get_valid_actions()
                if len(actions) != 0:
                    for action in actions:
                        self.game.set_state(leaf_data.game_state)
                        self.game.step(action)
                        new_node_data = MCTSNodeData(
                            value_est=0, 
                            num_visits=0, 
                            game_state=self.game.get_state(), 
                            prev_action=action,
                            player=self.game.get_curr_player())
                        self.tree.insert_node(new_node_data, leaf_node_idx)

                    leaf_node_idx = self.select(leaf_node_idx)
                    leaf_data = self.tree.get_node_data(leaf_node_idx)
                    idx_list.append(leaf_node_idx)

            # simulate and update values for every node visited
            value_est = self.simulate(leaf_data.game_state, curr_player)
            if self.value_network is not None:
                board_state = leaf_data.game_state[:9].reshape((1, 3, 3, 1))
                board_state = np.array(board_state, dtype=np.float32)
                if leaf_data.value_net_cache is None:
                    # cache value_net_cache for later
                    leaf_data.value_net_cache = self.value_network(board_state)
                    self.tree.update_node_data(leaf_node_idx, leaf_data)
                value_est = (0.5) * value_est + (0.5) * leaf_data.value_net_cache

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

        if curr_player == GameBase.Player.PLAYER1:
            best_idx = np.argmax(value_list)
        else:
            best_idx = np.argmin(value_list)
        best_action = self.tree.get_node_data(children[best_idx]).prev_action 
        return best_action

if __name__ == "__main__":
    from tictactoe import TicTacToe, TicTacToeHumanActor
    from game_base import run_game
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
    parser.add_argument(
        "--N",
        action="store",
        type=int,
        default=300,
        help="Number of expansions per search of MCTS tree")
    args = parser.parse_args()

    if args.player == 1:
        computer_player = TicTacToe.GridState.PLAYER2
    else:
        computer_player = TicTacToe.GridState.PLAYER1

    human_actor = TicTacToeHumanActor()
    ttt_search = TicTacToe()
    mcts_actor = MCTSActor(ttt_search, num_expansions=args.N)

    ttt = TicTacToe()
    human_actor.print_help()
    if args.player == 1:
        result = run_game(ttt, human_actor, mcts_actor)
    else:
        result = run_game(ttt, mcts_actor, human_actor)
    ttt.print_board()
    print("End Game Status: {}".format(result["game_status"].name))


