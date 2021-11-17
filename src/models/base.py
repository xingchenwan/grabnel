"""Base model"""
import dgl
import torch
import torch.nn as nn


class BaseGraphClassifier(nn.Module):
    """Base class."""

    def __init__(self, input_dim: int, number_of_labels: int, **kwargs):
        """

        Args:
            input_dim: Number of feature maps
            number_of_labels: Number of labels in the classification task
        """
        super(BaseGraphClassifier, self).__init__()
        self.input_dim = input_dim
        self.number_of_labels = number_of_labels

    def forward(self, graph: dgl.DGLGraph) -> torch.tensor:
        """

        Args:
            graph: a DGL graph with attributes stored in graph.ndata['node_attr']

        Returns: a torch tensor containing logits

        """
