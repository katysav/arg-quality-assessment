
from argparse import ArgumentParser

import dgl.data

from tqdm import tqdm

import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset
from dgl import DGLGraph

from model import GraphGATClassifier

import numpy as np


def collate(samples):
    '''Batch the graphs and their labels'''

    graphs, labels = map(list, zip(*samples))
    batched_graphs = dgl.batch(graphs)
    labels = torch.tensor(labels) - 1  # correct indexing for loss function
    return batched_graphs, torch.tensor(labels)


def evaluate(model, graphs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graphs)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(trainset, testset):
    '''Train and evaluate the model'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_train = DataLoader(
        trainset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

    model = GraphGATClassifier(5, 30, 3)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0
        for iter, (batched_graphs, labels) in enumerate(dataloader_train):
            logits = model(batched_graphs)
            loss = loss_func(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            acc = evaluate(model, batched_graphs, labels)
            epoch_acc += acc

        epoch_loss /= (iter + 1)
        epoch_acc /= (iter + 1)
        epoch_losses.append(epoch_loss)

        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f}".format(epoch, epoch_loss, epoch_acc))

    # Evaluation
    dataloader_test = DataLoader(
        testset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

    test_acc = 0
    for iter, (batched_graphs, labels) in enumerate(dataloader_test):
        acc = evaluate(model, batched_graphs, labels)
        test_acc += acc
    test_acc /= (iter + 1)
    print("Test accuracy {:.2%}".format(test_acc))


if __name__ == '__main__':
   
    parser = ArgumentParser(
        description='Main script')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    args = parser.parse_args()
    args = parser.parse_args()

    with open('graphs_scores_dict.pickle', 'rb') as file:
        dataset = pickle.load(file)

    graphs_train = dataset['dgl_graphs'][:250]
    acc_lables_train = dataset['acceptability_scores'][:250]
    graphs_test = dataset['dgl_graphs'][250:]
    acc_lables_test = dataset['acceptability_scores'][250:]


    trainset = []
    for graph, acc_label in zip(graphs_train, acc_lables_train):
        trainset.append((graph, acc_label))

    testset = []
    for graph, acc_label in zip(graphs_test, acc_lables_test):
        testset.append((graph, acc_label))


    main(trainset, testset)