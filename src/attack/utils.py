import random

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from scipy.sparse.csgraph import connected_components


def find_n_hop_neighbour(graph: dgl.DGLGraph, node_idx: int, n_hop: int, undirected=True,
                         exclude_self=True) -> torch.Tensor:
    """
    Given a node index, finds its n-hop neighbours in graph
    :param graph: dgl.DGLGraph: the graph object on which we find neighbours
    :param node_idx: the index of the node whose n-hop neighours we aim to find
    :param n_hop: int. number of hop distances to specify from node_idx
    :param undirected: bool. Whether to ignore any directedness in the graph (if the graph is already undirected, this
        flag does nothing)
    :param exclude_self: bool. whether to exclude node_idx itself from the list of neighbours.
    :return: a torch.Tensor of shape (n, ), where n is the total number of n-hop neighbours to node_index
     (including itself)
    """
    if undirected:
        graph = dgl.to_simple(deepcopy(graph))
    nodes = torch.tensor([node_idx])
    for i in range(n_hop):
        _, neighbours = graph.out_edges(nodes)
        nodes = torch.cat((nodes, neighbours))
        nodes = torch.unique(nodes)
    if exclude_self:
        nodes = nodes[nodes != node_idx]
    return nodes


def check_directed(graph: dgl.DGLGraph) -> bool:
    """
    Check whether a dgl graph is directed or undirected (or equivalently, every edge is bi-directed)
    :param graph:
    :return:
    """
    A = graph.adjacency_matrix().to_dense()
    return ~(A == torch.transpose(A, 0, 1)).all()


def get_allowed_nodes_k_hop(graph: dgl.DGLGraph, previous_edits, k_hop: int = 1):
    """
    For a graph, given a set of previous nodes, return the list of node indices within k_hop distance of previous edit
    nodes.
    When there is no previous edits,
    :param k_hop: int. the value of the hop distance to be considered neighbours. Default is 1.
    :param graph:
    :param previous_edits: the or list of edges editted previously.
    :return:
    """
    previous_edits = list(set(list(sum(previous_edits, ()))))   # flattens the list
    if len(previous_edits) == 0:
        return [i for i in range(graph.number_of_nodes())]
    allowed_nodes = torch.tensor([])
    for node in previous_edits:
        allowed_nodes = torch.cat((allowed_nodes, find_n_hop_neighbour(graph, node, k_hop, exclude_self=False)))
    return torch.unique(allowed_nodes).long()


def number_connected_components(dglgraph):
    return connected_components(dglgraph.adjacency_matrix(scipy_fmt="csr"))[0]


def random_sample_flip(graph: dgl.DGLGraph, budget: int,
                       prohibited_nodes: list = None,
                       prohibited_edges: list = None,
                       add_edge_only: bool = False,
                       remove_edge_only: bool = False,
                       n_hop: int = None,
                       allow_disconnected: bool = True,
                       preserve_disconnected_components: bool = False,
                       committed_edges = None,) -> set:
    """Perturb the graph using `budget` random flips.
    with_prior: bool.
        whether to select the nodes based on the prior information presented in the paper. Enabling this option has
        two effects:
            1. the probability of nodes being chosen will be inversely proportional to the sqrt(node degree). i.e. the
                lower degree nodes will have a higher chance of being selected.
            2. new perturbation candidates will be within 1-hop distance of previously suggested edits for at least
                one end-node (the other one is not affected)
    """
    assert (add_edge_only and remove_edge_only) is False, \
        'Either (or neither) add_edge_only and remove_edge_only can be True!'
    n_components = number_connected_components(graph)
    edges_to_flip = set()
    while len(edges_to_flip) < budget:
        patience = 100
        while patience > 0:
            all_nodes = range(graph.number_of_nodes())
            allowed_nodes = all_nodes

            if n_hop is None:
                u = np.random.choice(allowed_nodes, replace=False, )
                v = np.random.choice(allowed_nodes, replace=False,)
            else:
                u = np.random.choice(allowed_nodes, replace=False,)
                # for v can only be within the n-hop neighbours of u
                v_candidates = find_n_hop_neighbour(graph, u, n_hop)
                if v_candidates.shape[0] == 0:  # u is an isolated node (not supposed to happen for most of the time)
                    patience -= 1
                    continue
                v = np.random.choice(v_candidates)
            if u == v:
                patience -= 1
                continue
            if prohibited_nodes is not None:
                if u in prohibited_nodes or v in prohibited_nodes:
                    patience -= 1
                    continue
            if prohibited_edges is not None:
                if (u, v) in prohibited_edges or (v, u) in prohibited_edges:
                    patience -= 1
                    continue
            u, v = min(u, v), max(u, v)
            if graph.has_edges_between([u], [v])[0] and add_edge_only:
                patience -= 1
                continue
            if not graph.has_edges_between([u], [v])[0] and remove_edge_only:
                patience -= 1
                continue
            pert_graph = None
            if not allow_disconnected:
                if pert_graph is None: pert_graph = population_graphs(graph, [[(u, v)]], mode='flip')[0]
                nx_graph = pert_graph.to_networkx().to_undirected()
                if nx.number_connected_components(nx_graph) > 1:
                    patience -= 1
                    continue
            if preserve_disconnected_components:
                if pert_graph is None: pert_graph = population_graphs(graph, [[(u, v)]], mode='flip')[0]
                new_n_components = number_connected_components(pert_graph)
                if new_n_components != n_components:
                    patience -= 1
                    continue

            edges_to_flip.add((u, v))
            break
        if patience < 0:
            pass
    return edges_to_flip


def random_sample_rewire_swap(graph: dgl.DGLGraph, budget: int, rewire_only: bool = False,
                              swap_only: bool = False,
                              n_hop: int = None,
                              allow_disconnected: bool = False,
                              preserve_disconnected_components: bool = False,
                              ):
    """Rewire or swap 2 edges. Sample three nodes (u, v, w), where there is an existing edge on (u, v)
    if an edge is present on (u, w), this operation does swap: (u, v) <-> (u, w)
    otherwise, this operation does rewiring: (u, v) -> (u, w).
    Also note that swap operation does not change the graph at all (hence meaningless) if the problem has unweighted
        edges only.
     Note there are two edge flips per rewire budget"""
    edges_to_rewire = set()
    n_components = number_connected_components(graph)
    # return all nodes on which there are edges
    us, vs = graph.all_edges(order='eid')
    us, vs = us.numpy(), vs.numpy()
    all_edges = np.array([us, vs]).T
    while len(edges_to_rewire) < budget:
        patience = 100
        while patience > 0:
            # select (u, v) where existing edge is present
            idx = np.random.randint(all_edges.shape[0])
            (u, v) = all_edges[idx]
            # u = us[np.random.randint(0, len(us))]
            # v = vs[np.random.randint(0, len(vs))]
            if u == v:
                patience -= 1
                continue
            # select (u, w)
            if n_hop is None:
                w = np.random.randint(u, graph.num_nodes())
            else:
                w_candidates = find_n_hop_neighbour(graph, u, n_hop)
                if w_candidates.shape[0] == 0:
                    patience -= 1
                    continue
                w = np.random.choice(w_candidates)
            if u == w or v == w:
                patience -= 1
                continue
            # check whether (u, w) is an edge
            is_existing_edge = np.equal(all_edges, np.array([u, w])).all(1).any()
            if is_existing_edge:
                if rewire_only:
                    patience -= 1
                    continue
            else:
                if swap_only:
                    patience -= 1
                    continue
            # print(u,v,w)
            pert_graph = None
            if not allow_disconnected:
                if pert_graph is None:
                    pert_graph = population_graphs(graph, [[(u, v, w)]], mode='rewire')[0]
                nx_graph = pert_graph.to_networkx().to_undirected()
                if nx.number_connected_components(nx_graph) > 1:
                    patience -= 1
                    continue
            if preserve_disconnected_components:
                if pert_graph is None: pert_graph = population_graphs(graph, [[(u, v, w)]], mode='flip')[0]
                new_n_components = number_connected_components(pert_graph)
                if new_n_components != n_components:
                    patience -= 1
                    continue

            edges_to_rewire.add((u, v, w))
            break
        if patience <= 0:
            # print(f'Patience exhausted!')
            return edges_to_rewire
    return edges_to_rewire


def population_graphs(graph: dgl.DGLGraph, population: list, mode: str, ) -> list:
    """Takes the population and returns them in dgl.Graph format.
    graph: the base graph upon which we make edits
    population: a list of form
        [(n1, n2), (n3, n4) ... ] (flip mode) or
        [(n1, n2, n3), (n4, n5, n6)... ] (rewire mode)
    mode: 'flip' or 'rewire'. See descriptions below:
    for flip mode,
        Recall the population is a list of sets. Each set contains elements (u, v) where (u < v) representing flipping
        the undirected edge u ~ v. This method returns the population as a list of dgl.DGLGraph objects.
    for rewire mode,
        samples [(u, v, w), (u, v, w)...] where we rewire the edge u -> v to u -> w (if edge u -> w already exists,
        we swap the edges u -> v and u -> w)
    """
    perturbed_graphs = []
    is_edge_attributed = 'weight' in graph.edata.keys()
    for edge_to_edit in population:
        perturbed_graph = deepcopy(graph)
        for edge in edge_to_edit:
            if mode == 'rewire':
                (u, v, w) = edge
                if perturbed_graph.has_edges_between([u], [w])[0] and is_edge_attributed:
                    # swap for unweighted graph is meaningless
                    perturbed_graph.edges[u, w][0]['weight'] = torch.clone(graph.edges[u, v][0]['weight'])
                    perturbed_graph.edges[w, u][0]['weight'] = torch.clone(graph.edges[u, v][0]['weight'])
                    perturbed_graph.edges[u, v][0]['weight'] = torch.clone(graph.edges[u, w][0]['weight'])
                    perturbed_graph.edges[v, u][0]['weight'] = torch.clone(graph.edges[u, w][0]['weight'])
                else:  # rewire
                    flip_edge(perturbed_graph, u, v)  # delete the edge
                    flip_edge(perturbed_graph, u, w,
                              edata={'weight': graph.edges[u, w][0]['weight']} if is_edge_attributed else None)
            else:
                flip_edge(perturbed_graph, *edge)
        perturbed_graphs.append(perturbed_graph)
    return perturbed_graphs


def get_stages(max_budget, max_perturbation, mode='equidistant'):
    """
    During we attack, we may partition the total budget allocated. For example, for a total budget of 100 queries and
    if we allocate to up to 4 edge edits, we may partition the queries to 4 stages and only increase the number of edits
    added to the adjacency matrix iff we reach the next stage without a successful attack.

    Given the current query count, return its stage number
    :param n: current query number
    :param max_budget: maximum number of queries allowed
    :param max_perturbation: maximum number of edge edits allowed
    :param mode: 'equidistant': divide the total queries into stages with equal number of budgets. exp: use the
        successive halving type of division.
    :return:
    """
    assert mode in ['equidistant', 'exp']
    if mode == 'exp':
        # todo: implement the successive halving-style
        raise NotImplemented
    else:
        stages = np.linspace(0, max_budget, max_perturbation + 1)
    return stages


def get_device(gpu):
    """Return device string."""
    if torch.cuda.is_available() and gpu is not None:
        device = torch.device(f'cuda:{gpu}')
        print('Using', device, 'with cuda', torch.cuda.get_device_capability(device)[0], flush=True)
    else:
        device = torch.device('cpu')
        print('Using cpu', flush=True)
    return device


def setseed(seed):
    """Sets the seed for rng."""
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.random.manual_seed(seed)


def nettack_loss(logits, labels, target_class=None, **kwargs):
    """Implement the loss function in nettack
    target_class: if not None, the nettack loss will be the targeted loss w.r.t that label
    """

    def _single_eval(logit, label):
        logit = logit.flatten()
        # print(label)
        label = int(label)
        if logit.shape[0] <= 2:
            # binary problem -- convert the logit back to pseudo-probabilities (also, target class is not appliable here
            # since in this case the targeted attack is equivalent to untargeted attack
            logit = logit[0]
            class0 = torch.sigmoid(logit)
            diff_log = torch.log(class0) - torch.log(1. - class0)
            if label > 0:
                return -diff_log
            return diff_log
        else:
            if target_class is None:
                logit_ex_true = torch.cat([logit[:label], logit[label + 1:]])
                return torch.max(logit_ex_true - logit[label])
            return logit[target_class] - torch.max(logit)

    assert logits.shape[0] == labels.shape[0]
    if logits.shape[0] == 1:
        return _single_eval(logits, labels)
    else:
        losses = [_single_eval(logits[i], labels[i]) for i in range(logits.shape[0])]
        return torch.tensor(losses)


def nettack_loss_gunet(logits, labels, target_class=None, **kwargs):
    """Implement the loss function in nettack -- adapted to accommodate the slight difference in the GUNet API
    target_class: if not None, the nettack loss will be the targeted loss w.r.t that label.
    """

    def _single_eval(logit, label):
        logit = logit.flatten()
        # print(label)
        label = int(label)
        if target_class is None:
            logit_ex_true = torch.cat([logit[:label], logit[label + 1:]])
            return torch.max(logit_ex_true - logit[label])
        res = logit[target_class] - torch.max(logit)
        if torch.isnan(res):
            res = -100
        return res

    assert logits.shape[0] == labels.shape[0]
    if logits.shape[0] == 1:
        return _single_eval(logits, labels)
    else:
        losses = [_single_eval(logits[i], labels[i]) for i in range(logits.shape[0])]
        return torch.tensor(losses)


def classification_loss(is_binary):
    """Returns a loss function for classification tasks."""
    if is_binary:
        def loss_fn(x, y, **kwargs):
            return nn.functional.binary_cross_entropy_with_logits(x.squeeze(), y.float(), **kwargs)
    else:
        loss_fn = nn.functional.cross_entropy
    return loss_fn


def number_of_correct_predictions(predictions, labels, is_binary):
    """Sum of predictions with agree with labels. Predictions is given in logits."""
    if is_binary:
        return ((predictions.squeeze() > 0).float() == labels).sum()
    else:
        return (predictions.argmax(axis=1) == labels).sum()


def get_dataset_split(dataset):
    if 'IMDB' in dataset:  # for both IMDB-BINARY and IMDB-MULTI
        return 0.5, 0.3, 0.2
    elif dataset == 'er_graphs':
        return 0.89, 0.1, 0.01
    else:
        return 0.9, 0.05, 0.05  # changed from 0.9, 0.08, 0.02 -- 0.02 is too small it looks for many datasets.


def flip_edge(graph: dgl.DGLGraph, u: int, v: int, edata: dict = None, check_directness=False):
    """Flip the edge u ~ v in `graph`.

    This method assumes the graph is undirected. If the edge u ~ v exists it is deleted otherwise it is added.
    edata: if supplied, this specifies the edge feature
    """
    if check_directness:
        is_directed = check_directed(graph)
    else:
        is_directed = False

    if is_directed:
        if graph.has_edges_between([u], [v])[0]:
            edge_to_delete_id = graph.edge_ids(u, v)
            graph.remove_edges([edge_to_delete_id])
        else:
            graph.add_edges(u, v, data=edata)
    else:
        if graph.has_edges_between([u], [v])[0]:
            edge_to_delete_id = graph.edge_ids(u, v)
            edge_to_delete_reverse_id = graph.edge_ids(v, u)
            graph.remove_edges([edge_to_delete_id, edge_to_delete_reverse_id])
        else:
            graph.add_edges([u, v], [v, u], data=edata)


def correct_predictions(predictions: np.array, labels: np.array) -> int:
    """Returns number of predictions which are correctly assigned to their labels.

    Args:
        predictions: An (n x l) numpy array where n are the number of samples and l is the number of labels (or 1 if binary)
        labels: A 1D numpy array of predictions for each graph

    Returns:
        Number of correct predictions
    """
    assert isinstance(predictions, np.ndarray)
    assert isinstance(labels, np.ndarray)
    if predictions.shape[1] <= 2:  # binary classification
        predictions = predictions[:, 0]
        correct = (predictions > 0.0) == labels
    else:  # multiclass classification
        correct = np.argmax(predictions, axis=1) == labels
    return correct


def extrapolate_breakeven(historical_loss, using_last: int = 500):
    from sklearn.linear_model import LinearRegression
    if using_last is not None and using_last > 0:
        historical_loss = np.array(historical_loss).flatten()[-using_last:]
    else:
        historical_loss = np.array(historical_loss).flatten()
    # clear out and remove any nan and/or inf entries
    historical_loss = historical_loss[historical_loss == historical_loss]
    x = np.arange(len(historical_loss))
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), historical_loss)
    m, c = model.coef_, model.intercept_
    pt = -c / m
    offset = max(historical_loss.shape[0] - using_last, 0)
    return pt + offset
