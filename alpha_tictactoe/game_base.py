import numpy as np
import enum

class GameBase(object):
    class Status(enum.IntEnum):
        IN_PROGRESS = 0
        PLAYER1_WIN = 1
        PLAYER2_WIN = 2
        TIE = 3

    class Player(enum.IntEnum):
        PLAYER1 = 1
        PLAYER2 = 2

    def __init__(self):
        self.game_status = GameBase.Status.IN_PROGRESS
        self.curr_player = GameBase.Player.PLAYER1

    def reset(self):
        raise Exception("Not yet implemented.")

    def step(self, action):
        raise Exception("Not yet implemented.")


    def get_valid_actions(self):
        raise Exception("Not yet implemented.")

    def get_curr_player(self):
        return self.curr_player

    def get_game_status(self):
        return self.game_status

    def get_outcome(self, player):
        # Returns 0 for tie, 1 if player wins, -1 if player loses
        if self.game_status == GameBase.Status.IN_PROGRESS:
            raise Exception("Can't get outcome for a game that is not yet ended.")
        if self.game_status == GameBase.Status.TIE:
            return 0
        if self.game_status == player:
            return 1
        else:
            return -1

    def get_state(self):
        raise Exception("Not yet implemented.")

    def set_state(self, state):
        raise Exception("Not yet implemented.")

class GameActor(object):
    def get_action(self, game_state):
        """
        Must return a valid action.
        """
        raise Exception("Not yet implemented")

def run_game(game, actor1, actor2):
    data_dict = dict()
    data_dict["actions"] = []
    
    game.reset()

    while game.get_game_status() == GameBase.Status.IN_PROGRESS:
        curr_player = game.get_curr_player()
        if curr_player == GameBase.Player.PLAYER1:
            action = actor1.get_action(game.get_state())
        else:
            action = actor2.get_action(game.get_state())
        data_dict["actions"].append(action)
        game.step(action)

    data_dict["game_status"] = game.get_game_status()
    return data_dict



