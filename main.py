import argparse
import torch
import torchvision.datasets as dsets
import random
import numpy as np
import pandas as pd

import time
import matplotlib.pyplot as plt

from siarec.dataset import ASRTestDataset, ASRDataset
from siarec.model import SiameseNetwork
from siarec.engine import *
from siarec.utils import *

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5,
                help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                help='Number of images in each mini-batch')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                help='Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                help='Momentum')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                help='disabling CUDA training')
    parser.add_argument('--model', '-m', default='',
                help='Give a model to test')
    parser.add_argument('--data_path', default='./data',
                help='Give a dataset path')
    parser.add_argument('--train-plot', action='store_true', default=False,
                help='Plot train loss')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print("Args: %s" % args)

    
    train_df = pd.read_csv('./data/train/train.csv')
    test_df = pd.read_csv('./data/train/train.csv')
    train_iter = create_iterator(train_df.path, train_df.speaker, args.batch_size)

    # create pair dataset iterator
    train_dataloader = DataLoader(train_iter, batch_size=args.batch_size, shuffle=True)

    test_data = ASRTestDataset(test_df['path'], test_df['speaker'])
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


    # load model
    model = SiameseNetwork().cuda() if args.cuda else SiameseNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if len(args.model) == 0:
        train_loss = []
        for e in range(1, args.epoch+1):
            train_loss.extend(train_epoch(train_dataloader, model, optimizer, e, args))
        
        if args.train_plot:
            plt.gca().cla()
            plt.plot(train_loss, label="train loss")
            plt.legend()
            plt.draw()
            plt.savefig('train_loss.png')
            plt.gca().clear()
    
    else:
        print("Testing. . .")
        saved_model = torch.load(args.model)
        model = SiameseNetwork()
        model.load_state_dict(saved_model)
        if args.cuda:
            model.cuda()

        numpy_pred, numpy_labels = test_epoch(test_dataloader, model, args)
        breakpoint()


if __name__ == '__main__':
    main()