#!/usr/bin/python3

import dgl
#import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GATConv, GraphConv
from dgl.nn.pytorch.glob import MaxPooling

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGATClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_classes):

        super(GraphGATClassifier, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, 1)
        self.layer2 = GATConv(hidden_dim, hidden_dim, 1)
        self.classify = nn.Linear(hidden_dim, num_classes)


    def forward(self, g):

        h = g.in_degrees().view(-1, 1).float()  # node features == node degree
        h = F.elu(self.layer1(g, h)).flatten(1)
        h = F.elu(self.layer2(g, h)).flatten(1)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        return self.classify(hg)
