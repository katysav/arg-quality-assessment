from argparse import ArgumentParser
from collections import Counter
import pickle
import random

import dgl.data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import wandb

from model import GraphGATClassifier


def collate(samples):
    """Batch the graphs and their labels"""

    graphs, labels = map(list, zip(*samples))
    batched_graphs = dgl.batch(graphs)
    labels = torch.tensor(labels) - 1  # correct indexing for loss function
    return batched_graphs, torch.tensor(labels)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def compute_acc(model, graphs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graphs)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def calculate_class_weights(labels):
    """Calculate weights for imbalanced classes
    """
    class_counts = sorted(Counter(labels).items())
    num_items = [x[1] for x in class_counts]
    weights = [min(num_items)/x for x in num_items]

    return torch.tensor(weights)


def handle_imbalance(dataset, minority_class):
    """Delete the minority class, assign its
    label to the nearest majority class
    """
    for i, l in enumerate(dataset):
            if l == minority_class:
                dataset[i] = 2
    return dataset


def split_sets(dataset_x, dataset_y, random_state):

    # Divide into train and test sets: 0.8/0.2
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_x, dataset_y, test_size=0.2, random_state=random_state)
    # Divide the test set into validation and test 0.1/0.1
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state)


    return X_train, X_val, X_test, y_train, y_val, y_test


def main(train, val, test, class_weights):

    """Train and evaluate the model"""

    wandb.init(project="arg-qual", tags=[args.node_feat, args.quality_dim, str(args.epochs), str(args.lr), "val=test"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_train = DataLoader(
        trainset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

    dataloader_val = DataLoader(
            valset,
            batch_size=10,
            collate_fn=collate,
            drop_last=False,
            shuffle=True)

    in_dim = trainset[0][0].ndata['x'].shape[1]  # define feature dim
    model = GraphGATClassifier(in_dim, 10, 2)
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    set_seed(args)

    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for iter, (batched_graphs, labels) in enumerate(dataloader_train):
            logits = model(batched_graphs)
            loss = loss_func(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            acc = compute_acc(model, batched_graphs, labels)
            epoch_acc += acc

        epoch_loss /= (iter + 1)
        epoch_acc /= (iter + 1)
        epoch_losses.append(epoch_loss)

        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        wandb.log({'loss': epoch_loss})

        # Evaluate during training
        val_acc = 0
        for iter, (batched_graphs, labels) in enumerate(dataloader_val):
            acc = compute_acc(model, batched_graphs, labels)
            val_acc += acc
        val_acc /= (iter + 1)
        print("Validation accuracy {:.2%} at epoch {}".format(val_acc, epoch))
        wandb.log({'val_acc': val_acc})

    # Evaluation
    dataloader_test = DataLoader(
        testset,
        batch_size=10,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

    test_acc = 0
    for iter, (batched_graphs, labels) in enumerate(dataloader_test):
        acc = compute_acc(model, batched_graphs, labels)
        test_acc += acc
    test_acc /= (iter + 1)
    print("Test accuracy {:.2%}".format(test_acc))
    wandb.log({'test_acc': test_acc})


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Main script")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=80,
                        help="number of training epochs")
    parser.add_argument("--seed", type=int, default=42, 
                        help="random seed for initialization")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="weight decay")
    parser.add_argument("--quality_dim", choices=["rel", "suf", "acc", "cog"],
                        default="cog",
                        help="Argument qulity dimension to evaluate: \
                        relevance, sufficiency, acceptability, cogency.")
    parser.add_argument("--node_feat", choices=["rand", "glove", "bert"],
                        default="rand",
                        help="Graph node features: random vectors, Glove embeddings, \
                        BERT sentence embeddings")
    args = parser.parse_args()

    with open("data/graphs_scores_dict.pickle", "rb") as file:
        dataset = pickle.load(file)

    # Select quality dimension to evaluate
    if args.quality_dim == "rel":
        labels = handle_imbalance(dataset["relevance_scores"], 1)
        for i, n in enumerate(labels):  # needed to fix an index error
            if n == 2:
                labels[i] = 1
            elif n == 3:
                labels[i] = 2
    elif args.quality_dim == "suf":
        labels = handle_imbalance(dataset["sufficiency_scores"], 3)
    elif args.quality_dim == "acc":
        labels = handle_imbalance(dataset["acceptability_scores"], 3)
    elif args.quality_dim == "cog":
        labels = handle_imbalance(dataset["cogency"], 3)


    # Select features nodes
    if args.node_feat == "rand":
        graphs_train, graphs_val, graphs_test, labels_train, labels_val, labels_test = split_sets(dataset["dgl_graphs_rand"], labels, 1)
    elif args.node_feat == "glove":
        graphs_train, graphs_val, graphs_test, labels_train, labels_val, labels_test = split_sets(dataset["dgl_graphs_glove"], labels, 1)
    elif args.node_feat == "bert":
        graphs_train, graphs_val, graphs_test, labels_train, labels_val, labels_test = split_sets(dataset["dgl_graphs_bert"], labels, 1)


    trainset = []
    for graph, label in zip(graphs_train, labels_train):
        trainset.append((graph, label))

    valset = []
    for graph, label in zip(graphs_val, labels_val):
        valset.append((graph, label))

    testset = []
    for graph, label in zip(graphs_test, labels_test):
        testset.append((graph, label))

    class_weights = calculate_class_weights(labels_train)

    main(trainset, testset, testset, class_weights)
