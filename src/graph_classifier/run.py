
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
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def evaluate(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(trainset):
    '''Train and evaluate the model'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = MiniGCDataset(200, 10, 20)
    dataloader = DataLoader(
        trainset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)
    running_loss = 0
    total_iters = len(dataloader)

    model = GraphGATClassifier(1, 16, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for iter, (batched_graph, labels) in enumerate(dataloader):
            logits = model(batched_graph)
            loss = loss_func(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        epoch_losses.append(epoch_loss)

        acc = evaluate(model, batched_graph, labels)
        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f}".
                format(epoch, epoch_loss, acc))

    #acc = evaluate(model, features, labels, test_mask)
    #print("Test accuracy {:.2%}".format(acc))


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
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    args = parser.parse_args()

    with open('graphs_scores_dict.pickle', 'rb') as file:
        dataset = pickle.load(file)

    graphs_train = dataset['node_feature_matrix'][:100]

    acc_train_lables = dataset['acceptability_scores'][:100]

    data = []
    for graph, acc_label in zip(graphs_train, acc_train_lables):
        data.append((graph, acc_label))

    main(data)