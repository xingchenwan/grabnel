import torch.nn as nn
import dgl
import torch
import torch.nn.functional as F
from .base import BaseGraphClassifier
from pytorch_structure2vec.s2v_lib.embedding import EmbedMeanField, EmbedLoopyBP
#from pytorch_structure2vec.graph_classification.util import S2VGraph
import numpy as np
import networkx as nx


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None):
        self.num_nodes = len(g)
        self.node_tags = node_tags
        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.label = label
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

    def to_networkx(self):
        edges = np.reshape(self.edge_pairs, (self.num_edges, 2))
        g = nx.Graph()
        g.add_edges_from(edges)
        return g


class S2VClassifier(BaseGraphClassifier):

    def __init__(self, input_dim, number_of_labels, latent_dim=64, embedding_output_dim=64, max_lv=2, gm='mean_field'):
        super(S2VClassifier, self).__init__(input_dim, number_of_labels)

        # select model class
        if gm == 'mean_field':
            model = EmbedMeanField
        elif gm == 'loopy_bp':
            model = EmbedLoopyBP

        print('input_dim', input_dim, 'max_lv', max_lv)
        # initialise embedding model
        self.s2v = model(latent_dim=latent_dim, output_dim=0, num_node_feats=input_dim,
                         num_edge_feats=0, max_lv=max_lv)

        # MLP layer
        self.mlp = MLP(input_size=embedding_output_dim, hidden_size=32, number_of_labels=number_of_labels)

    # patched by xingchen to avoid the batching / unbatching error in s2v
    # def forward(self, graph: dgl.DGLGraph) -> torch.tensor:
    #     s2v_graphs, features = self.prepare_batch(graph)
    #     embedding = self.s2v(s2v_graphs, features, None)
    #     return self.mlp(embedding)

    def forward(self, graphs) -> torch.Tensor:
        s2v_graphs, features = self.prepare_batch(graphs)
        embedding = self.s2v(s2v_graphs, features, None)
        return self.mlp(embedding)

    # @staticmethod
    # def prepare_batch(graph_batch:  dgl.DGLGraph) -> (list, torch.tensor):
    #     features = graph_batch.ndata['node_attr']
    #     graphs = dgl.unbatch(graph_batch)
    #     s2v_graphs = []
    #     for graph in graphs:
    #         label = number_connected_components(graph)
    #         g = graph.to_networkx()
    #         s2v_graphs.append(S2VGraph(g, label))
    #     return s2v_graphs, features
    #
    @staticmethod
    def prepare_batch(graphs) -> (list, torch.tensor):
        if not isinstance(graphs, list):
            graphs = [graphs]       # one element
        features = torch.cat([graph.ndata['node_attr'] for graph in graphs])
        # graphs = dgl.unbatch(graph_batch)
        s2v_graphs = []
        for graph in graphs:
            label = number_connected_components(graph)
            g = graph.to_networkx()
            s2v_graphs.append(S2VGraph(g, label))
        return s2v_graphs, features


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, number_of_labels):
        super(MLP, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.last_weights = nn.Linear(hidden_size, number_of_labels)

    def forward(self, x):
        x = self.h1_weights(x)
        x = F.relu(x)
        x = self.last_weights(x)
        #x = F.log_softmax(x, dim=1)
        return x


from scipy.sparse.csgraph import connected_components

def number_connected_components(dglgraph):
    return connected_components(dglgraph.adjacency_matrix(scipy_fmt="csr"))[0]