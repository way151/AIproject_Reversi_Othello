import os
import torch

from model.resnet import ResNet
from game.board import OthelloGame

from game.players import RandomPlayer
from game.players import OneStepLookaheadPlayer


class Config():
    # Overall setting
    name = 'othello_resnet'
    game = OthelloGame
    nnet = ResNet
    use_multiprocessing = True


    # RL Training
    numIters = 1000
    numEps = 100
    tempThreshold = 15 # 30 in paper
    updateThreshold = 0.55 # 0.55 in AlphaGoZero, not in AlphaZero (used continuous updates, no selection)
    maxlenOfQueue = 200000
    numMCTSSims = 25 # 1600 is AlphaGoZero, 800 in AlphaZero
    arenaCompare = 40 # number of games of self play to choose previous or current nnet (400 in paper)
    cpuct = 1
    numItersForTrainExamplesHistory = 20


    # Hardware
    num_workers = 4
    cuda = torch.cuda.is_available() # use cuda if available


    # Model Architecture
    dropout = 0.3
    num_channels = 512
    # nnet_kwargs = {'num_channels':num_channels, 'dropout':dropout}
    nnet_kwargs = {}


    # Model Training
    epochs = 10 # num of epochs for single train iteration (not in AlphaZero, use continuous training)
    batch_size = 64 # 4096 in paper
    lr = 0.001
    optimizer = torch.optim.Adam
    optimizer_kwargs = {'betas': (0.9, 0.999), 'weight_decay':0.001}
    lr_scheduler = torch.optim.lr_scheduler.StepLR
    lr_scheduler_kwargs = {'step_size':1, 'gamma':0.967}


    # Metrics
    metric_opponents = [RandomPlayer, OneStepLookaheadPlayer]
    metricArenaCompare = 20 # number of games to play against metric opponent


    # Model Loading
    checkpoint = os.path.join('saved/', name)
    load_model = False # load model
    load_model_file = (checkpoint, 'best.pth.tar')
    load_train_examples = False # load training examples
    load_folder_file = (checkpoint,'best.pth.tar') # file to training examples


    # Logging
    log_dir = 'saved/runs'
    tensorboardX = True
