"""
Based on
https://github.com/suragnair/alpha-zero-general
"""


import numpy as np
from functools import partial
import torch.multiprocessing as mp
from tqdm import tqdm

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

        # list to store results of games for elo (1 for player1 wins, -1 player2 wins)
        self.results = []

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [None, self.player1, self.player2]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            assert valids[action] > 0, "action {} is not valid".format(action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        bar = tqdm(desc='Arena.playGames', total=num)
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            self.results.append(gameResult)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            bar.update()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            self.results.append(-gameResult)
            if gameResult==-1:
                oneWon+=1
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            bar.update()

        bar.close()

        return oneWon, twoWon, draws

    def get_results(self):
        return self.results


"""
Multiprocessing
"""
import os
class ArenaMP(Arena):
    """
    Arena class that utilizes multiprocessing.
    Note: Use non-human players only
    """

    @staticmethod
    def playGame(player1, player2, game, display, verbose=False, _=None):
        """
        Executes one episode of a game.
        Params:
            player1
            player2
            game
            display: function to display the board
            verbose: bool to display everything
            _: ignore, used only for pool imap
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [None, player1, player2]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                display(board)

            action = players[curPlayer](game.getCanonicalForm(board, curPlayer))

            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer),1)

            assert valids[action] > 0, "action {} is not valid".format(action)
            board, curPlayer = game.getNextState(board, curPlayer, action)
        if verbose:
            assert(display)
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded(board, 1)))
            display(board)
        return game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False, num_workers=mp.cpu_count()):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Params:
            num: number of games to play
            verbose: bool to show everything
            num_workers: number of workers to use for multiprocessing
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        bar = tqdm(desc='Arena.PlayGames', total=num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        with mp.Pool(processes=num_workers) as pool:
            for gameResult in pool.imap_unordered(partial(self.playGame, self.player1,
                                                    self.player2, self.game, self.display, verbose), range(num)):
                self.results.append(gameResult)
                if gameResult==1:
                    oneWon+=1
                elif gameResult==-1:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                bar.update()

            self.player1, self.player2 = self.player2, self.player1

            for gameResult in pool.imap_unordered(partial(self.playGame, self.player1, self.player2, self.game, self.display, verbose), range(num)):
                # append negative result, because players switched
                self.results.append(-gameResult)
                if gameResult==-1:
                    oneWon+=1
                elif gameResult==1:
                    twoWon+=1
                else:
                    draws+=1
                # bookkeeping + plot progress
                bar.update()

        bar.close()

        return oneWon, twoWon, draws
