
from Arena import Arena
from game.board import display
from game.board import OthelloGame

from game.players import HumanOthelloPlayer
# from players.MCTSPlayer import MCTSPlayer
from game.players import NNetPlayer
from game.players import GAPlayer
from PytorchNNet import NNetWrapper


from config import Config


if __name__ == '__main__':
    game = OthelloGame()
    config = Config()

    # rp = RandomPlayer(game).play
    hp = GAPlayer(game).play
    # mctsp = MCTSPlayer(game, config).play

    nn = NNetWrapper(game, config)
    ckpt = ('saved','best.pth.tar')
    # ckpt = ('./trained/othello_resnet','best.pth.tar')
    nn.load_checkpoint(ckpt[0], ckpt[1])
    nnp = NNetPlayer(game, nn, config).play

    #nn2 = NNetWrapper(game, config)
    #ckpt2 = ('saved','checkpoint_8.pth.tar')
    #nn2.load_checkpoint(ckpt2[0], ckpt2[1])
    #nnp2 = NNetPlayer(game, nn2, config).play


    arena = Arena(hp, nnp, game, display=display)
    out = arena.playGames(50, verbose=True)
    print(out)