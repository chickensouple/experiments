import numpy as np
import enum
from game_base import GameBase


class TicTacToe(GameBase):
    """
    Simple Tic Tac Toe game
    
    Attributes:
        board (np array): (3, 3) array representing board state
        -1 is empty, 0 is player 0, 1, is player 1
        curr_player (int): current player (0 or 1).
        winner (int): indicates which player is the winner,
        or -1 if the game is a tie
        If game has not ended, this will be None
    """
    class GridState(enum.IntEnum):
        EMPTY = 0
        PLAYER1 = 1
        PLAYER2 = 2

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets game state to start of a new game
        """
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.curr_player = GameBase.Player.PLAYER1
        self.game_status = GameBase.Status.IN_PROGRESS

    def get_valid_actions(self):
        """
        Get current valid actions for the current player
        
        Returns:
            list: list of (x, y) tuples of game positions that can be played
        """
        if self.game_status != GameBase.Status.IN_PROGRESS:
            return []

        x_vals, y_vals = np.where(self.board == TicTacToe.GridState.EMPTY)
        valid_actions = list(zip(x_vals, y_vals))
        return valid_actions

    def _check_valid_action(self, action):
        """
        @brief      checks if an action is valid

        @param      action  tuple of two integers. each from the set {0, 1, 2}        
        """
        row, col = action
        if row < 0 or row > 2:
            return False
        if col < 0 or col > 2:
            return False
        val = self.board[row, col]
        return (val == TicTacToe.GridState.EMPTY)

    def get_state(self):
        state = np.zeros(11, dtype=np.int8)
        state[:9] = np.reshape(self.board, (9))
        state[9] = self.curr_player
        state[10] = self.game_status
        return state

    def set_state(self, state):
        self.board = np.copy(np.reshape(state[:9], (3, 3)))
        self.curr_player = state[9]
        self.game_status = state[10]

    def step(self, action):
        """
        takes an action for the current player
        
        Args:
            action (tuple): (x, y) location of action on the board  
        """
        assert(type(action) is tuple)

        if self.game_status != GameBase.Status.IN_PROGRESS:
            return

        if not self._check_valid_action(action):
            self.print_board()
            raise Exception("Invalid Action")

        if self.curr_player == GameBase.Player.PLAYER1:
            self.board[action] = TicTacToe.GridState.PLAYER1
        else:
            self.board[action] = TicTacToe.GridState.PLAYER2


        if self.curr_player == GameBase.Player.PLAYER1:
            self.curr_player = GameBase.Player.PLAYER2
        else:
            self.curr_player = GameBase.Player.PLAYER1

        self.game_status = self._check_winner()


    def print_board(self):
        """
        Prints current state of game board
        player 1 is indicated by 'O'
        player 2 is indicated by 'X'
        """
        print('    0   1   2')
        print('  -------------')

        print_dict = {
            TicTacToe.GridState.EMPTY: ' ', 
            GameBase.Player.PLAYER1: 'O', 
            GameBase.Player.PLAYER2: 'X'}
        for i in range(3):
            print_str = str(i) + ' | '
            for j in range(3):
                print_str += print_dict[self.board[i, j]]
                print_str += ' | '
            print(print_str)
            print('  -------------')

    def _check_winner(self):
        for i in range(3):
            # check rows
            row = self.board[i, :]
            if np.all(row == GameBase.Player.PLAYER1):
                return GameBase.Status.PLAYER1_WIN
            elif np.all(row == GameBase.Player.PLAYER2):
                return GameBase.Status.PLAYER2_WIN

            # check columns
            col = self.board[:, i]
            if np.all(col == GameBase.Player.PLAYER1):
                return GameBase.Status.PLAYER1_WIN
            elif np.all(col == GameBase.Player.PLAYER2):
                return GameBase.Status.PLAYER2_WIN

        # check diagonals
        diag = np.diag(self.board) 
        if np.all(diag == GameBase.Player.PLAYER1):
            return GameBase.Status.PLAYER1_WIN
        elif np.all(diag == GameBase.Player.PLAYER2):
            return GameBase.Status.PLAYER2_WIN

        diag = np.diag(np.fliplr(self.board))
        if np.all(diag == GameBase.Player.PLAYER1):
            return GameBase.Status.PLAYER1_WIN
        elif np.all(diag == GameBase.Player.PLAYER2):
            return GameBase.Status.PLAYER2_WIN

        # check for tie
        if np.all(self.board != TicTacToe.GridState.EMPTY):
            return GameBase.Status.TIE
        return GameBase.Status.IN_PROGRESS


if __name__ == "__main__":
    ttt = TicTacToe()
    print("Actions are input as a 2 numbers seperated by a space, indicating the row and column.")
    
    while ttt.get_game_status() == GameBase.Status.IN_PROGRESS:
        ttt.print_board()
        print("Player {}'s turn.".format(ttt.get_curr_player()))
        action = input("Enter an action (row, col): ")
        try:
            action = action.split(" ")
            action = tuple([int(idx) for idx in action])

            ttt.step(action)
        except Exception as e:
            print(e)
            print("Not an valid action. Try again.")
            continue
    ttt.print_board()
    print("End Game Status: {}".format(ttt.get_game_status().name))

