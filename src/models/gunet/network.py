import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gunet.utils.ops import GCN, GraphUnet, Initializer, norm_g

from src.models.base import BaseGraphClassifier
import dgl
import numpy as np


def parse_dgl_graph(graph):
    """Parse the dgl graph"""
    if isinstance(graph, list):
        graphs = graph
    elif isinstance(graph, dgl.DGLGraph):
        try:
            graphs = dgl.unbatch(graph)
        except RuntimeError:
            graphs = [graph]
    res = []
    for graph in graphs:
        N_nodes = graph.number_of_nodes()
        u, v = graph.all_edges(order='eid')
        wadj = np.zeros((N_nodes, N_nodes))
        if 'weight' in graph.edata.keys():
            wadj[u.numpy(), v.numpy()] = graph.edata['weight'].numpy()
        else:
            wadj[u.numpy(), v.numpy()] = 1.
        adj = torch.tensor(wadj).float().unsqueeze(0)
        if 'node_attr' in graph.ndata.keys():
            node_feat = graph.ndata['node_attr'].unsqueeze(0)
        else:
            node_feat = graph.in_degrees().numpy().reshape(-1, 1)
        mask = torch.ones(1, N_nodes, dtype=torch.uint8)
        res.append([node_feat, adj, mask, None, {'N_nodes': torch.zeros(1, 1) + N_nodes}])
    return res


class GUNet(BaseGraphClassifier):
    def __init__(self, net, input_dim=3, number_of_labels=2):
        """Wrapper around GUNet into a BaseGraphClassifier. Used as a victim for the BO attack"""
        super().__init__(input_dim, number_of_labels)
        self.net = net
        self.net.eval()

    def forward(self, graph: dgl.DGLGraph):
        parsed_data = parse_dgl_graph(graph)
        res = torch.empty(size=(len(parsed_data), self.number_of_labels))
        for i, g in enumerate(parsed_data):
            res[i] = self.net.get_logit(g[1], g[0])
        return res

    def is_correct(self, graph: dgl.DGLGraph, labels):
        parsed_data = parse_dgl_graph(graph)
        acc = torch.empty(size=(len(parsed_data), 1))
        for i, g in enumerate(parsed_data):
            acc[i] = self.net(g[1], g[0], labels[i])[1]
        return acc


class GNet(nn.Module):
    """The original GNet module. Used for model training"""
    def __init__(self, in_dim, n_classes, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n)
        self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.h_dim)
        self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        self.out_drop = nn.Dropout(p=args.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        hs = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels)

    def get_logit(self, gs, hs):
        """Get logits in eval mode"""
        with torch.no_grad():
            hs = self.embed(gs, hs)
            return self.classify(hs)

    def embed(self, gs, hs):
        o_hs = []
        for g, h in zip(gs, hs):
            h = self.embed_one(g, h)
            o_hs.append(h)
        hs = torch.stack(o_hs, 0)
        return hs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        hs = self.g_unet(g, h)
        h = self.readout(hs)
        return h

    def readout(self, hs):
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc
