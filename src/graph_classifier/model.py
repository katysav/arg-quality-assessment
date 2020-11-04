#!/usr/bin/python3

import dgl
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.glob import MaxPooling

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGATClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_classes, p):

        super(GraphGATClassifier, self).__init__()
        self.dropout = nn.Dropout(p)
        self.layer1 = GATConv(in_dim, hidden_dim, 1, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_dim, hidden_dim, 1, allow_zero_in_degree=True)
        self.layer3 = GATConv(hidden_dim, hidden_dim, 1, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, num_classes)


    def forward(self, g):

        h = g.ndata['x']  # graph features
        h = F.elu(self.layer1(g, h)).flatten(1)
        h = F.elu(self.layer2(g, h)).flatten(1)
        h = F.elu(self.layer3(g, h)).flatten(1)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
        hg = self.dropout(hg)
        return self.classify(hg)