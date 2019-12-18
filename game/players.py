import numpy as np
from model.mcts import MCTS 
class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class OneStepLookaheadPlayer():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, verbose=False):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))

        return ret_move

class NNetPlayer():
    """
    Wrapper for neural network + MCTS player. Used for multiprocessing since
    we can't pickle lambda functions
    Params:
        game
        nnet: neural net to use
        args: config
    """
    def __init__(self, game, nnet, args):
        self.mcts = MCTS(game, nnet, args)

    def play(self, x, temp=0):
        return np.argmax(self.mcts.getActionProb(x, temp=temp))


class DummyNNet():
        """
        Dummy Neural Network: returns 0 for everything
        """
        def __init__(self, game, args):
            self.game = game
            self.args = args
            self.player_num = 1

            self.action_size = self.game.getActionSize()

        def predict(self, board):
            pi = np.zeros(self.action_size)

            v = self.game.getGameEnded(board, self.player_num)

            valid_moves = self.game.getValidMoves(board, self.player_num)
            for move, valid in enumerate(valid_moves):
                if not valid: continue
                if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                    pi[move] = 1
                elif -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                    pi[move] = -1

            return pi, v

class MCTSPlayer():
    """
    Wrapper for neural network + MCTS player. Used for multiprocessing since
    we can't pickle lambda functions
    Params:
        game
        nnet: neural net to use
        args: config
    """
    def __init__(self, game, args):
        dummynnet = DummyNNet(game, args)
        self.mcts = MCTS(game, dummynnet, args)

    def play(self, x, temp=0):
        return np.argmax(self.mcts.getActionProb(x, temp=temp))
