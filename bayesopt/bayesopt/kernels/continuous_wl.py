import dgl
from typing import List
import numpy as np
from sklearn.preprocessing import scale


class ContinuousWeisfeilerLehman:
    def __init__(self, h: int = 1, node_feat_name='node_attr1', edge_feat_name='weight'):
        """
        The extension of WL to graphs with continuous node attributes and/or edge attributes. This module uses the
            extension in Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B., & Borgwardt, K. (2019). Wasserstein
            Weisfeiler-Lehman graph kernels. ArXiv, (1), 1–19 that extends the WL-categorical to WL-continuous search
            space.
        """
        self.node_feat_name = node_feat_name  # todo!
        self.edge_feat_name = edge_feat_name
        self.h = h
        self.X = []
        self.train_feature_mean, self.train_feature_std = None, None

    def _preprocess_graphs(self, G: List[dgl.DGLGraph]):
        """Preprocess the list of graphs to obtain a 3-tuple of  node_features, adjacency matrices and number of nodes
        """
        node_features = []
        adj_mats = []
        n_nodes = []
        for i, g in enumerate(G):
            nnode, node_feat, adj = parse_dgl_graph(g, node_feat_name=self.node_feat_name, edge_feat_name=self.edge_feat_name)
            adj_mats.append(adj)
            node_features.append(node_feat)
            n_nodes.append(nnode)
        return node_features, adj_mats, n_nodes

    def parse_input(self, X, return_label=False, train_mode=True):
        if not train_mode and self.train_feature_mean is None:
            raise ValueError("Eval mode enabled but self.X has length = 0.")
        node_features, adj_mat, n_nodes = self._preprocess_graphs(X)
        node_features_concat = np.concatenate(node_features, axis=0)
        if train_mode:
            self.train_feature_mean = np.mean(node_features_concat, axis=0)
            self.train_feature_std = np.std(node_features_concat, axis=0)
        node_features_data = (node_features_concat - self.train_feature_mean) / self.train_feature_std

        # for complete graphs, this might cause node_features to be all zero. If that is the case, rescale node_features to all 1 so that features won't get zeroed out
        # node_features_data += 1e-2
        splits_idx = np.cumsum(n_nodes).astype(int)
        node_features_split = np.vsplit(node_features_data, splits_idx)
        node_features = node_features_split[:-1]

        # Generate the label sequences for h iterations
        n_graphs = len(node_features)
        label_sequences = []
        for i in range(n_graphs):
            graph_feat = []

            for it in range(self.h + 1):
                if it == 0:
                    graph_feat.append(node_features[i])
                else:
                    adj_cur = adj_mat[i] + np.identity(adj_mat[i].shape[0])
                    adj_cur = _create_adj_avg(adj_cur)
                    # print(adj_cur)
                    # exit()

                    np.fill_diagonal(adj_cur, 0)
                    graph_feat_cur = 0.5 * (np.dot(adj_cur, graph_feat[it - 1]) + graph_feat[it - 1])
                    graph_feat.append(graph_feat_cur)
            # first concatenate across different h:
            graph_feat = np.concatenate(graph_feat, axis=0).flatten()
            label_sequences.append(graph_feat)
        label_sequences = np.asarray(label_sequences)
        return label_sequences

    def fit(self, *args, **kwargs):
        """Alias for fit_transform for consistency of API."""
        return self.fit_transform(*args, **kwargs)

    def fit_transform(self, X, y=None):
        self.X = self.parse_input(X, y, train_mode=True)
        return self.X

    def transform_parse(self, X):
        return self.parse_input(X, train_mode=False)

    def transform(self, X):
        """Alias for transform_parse"""
        return self.transform_parse(X)


def _create_adj_avg(adj_cur):
    """
    create adjacency
    """
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg != 1] -= 1

    deg = 1 / deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T
    return adj_cur


def parse_dgl_graph(graph: dgl.DGLGraph, node_feat_name: str = None, edge_feat_name: str = None):
    if node_feat_name is None:
        node_feat_name = 'node_attr'
    if edge_feat_name is None:
        edge_feat_name = 'weight'
    N_nodes = int(graph.number_of_nodes())
    u, v = graph.all_edges(order='eid')
    wadj = np.zeros((N_nodes, N_nodes))
    if edge_feat_name in graph.edata.keys():
        wadj[u.numpy(), v.numpy()] = graph.edata[edge_feat_name].numpy()
    else:
        wadj[u.numpy(), v.numpy()] = 1.
        # wadj = graph.adjacency_matrix().to_dense().numpy()
    # adj = wadj[np.newaxis, :]
    if node_feat_name in graph.ndata.keys():
        node_feat = graph.ndata[node_feat_name].numpy()
    else:
        # print(f'Node feature {node_feat_name} does not exist! Using node degree')
        node_feat = graph.in_degrees().numpy().reshape(-1, 1)
    # print(N_nodes, node_feat.shape, wadj.shape)
    return N_nodes, node_feat, wadj
