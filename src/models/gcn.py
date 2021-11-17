"""GCN based classification model."""

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import MaxPooling

from .base import BaseGraphClassifier


class GCNGraphClassifier(BaseGraphClassifier):
    """A GCN based graph classifier. Outputs are logits.

    This model is based off the ReWatt model.
    """

    def __init__(self, input_dim: int, number_of_labels: int, hidden_dim: int = 16):
        super(GCNGraphClassifier, self).__init__(input_dim, number_of_labels)
        self.hidden_dim = hidden_dim
        self.graph_conv1 = GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        self.graph_conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.graph_conv3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.pooling = MaxPooling()
        if number_of_labels == 2:
            self.MLP = nn.Linear(16, 1)
        else:
            self.MLP = nn.Linear(16, number_of_labels)

    def forward(self, graph: dgl.DGLGraph, edge_weight=None) -> torch.tensor:
        """Produce logits.
        Edit: added edge_weight options (for dgl > 0.6.0)
        """
        x = graph.ndata['node_attr']
        x = torch.relu(self.graph_conv1(graph, x, edge_weight=edge_weight))
        x = torch.relu(self.graph_conv2(graph, x, edge_weight=edge_weight))
        x = torch.relu(self.graph_conv3(graph, x, edge_weight=edge_weight))
        x = self.pooling(graph, x)
        x = self.MLP(x)
        return x


