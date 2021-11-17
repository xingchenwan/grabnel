"""Generate partitions of erdos-renyi graphs (following methodologies of Dai et al 2018).

The data is stored as a list of tuples where each tuple is of the form (x, y) where x is a DGLGraph and y is a torch
tensor containing the label."""

import argparse
import os
import pickle
from os.path import join

import dgl
import networkx as nx
import numpy as np
import torch
import tqdm
from utils import setseed

parser = argparse.ArgumentParser()
parser.add_argument('--min_n', type=int, default=90, help='Minimum number of nodes in each component.')
parser.add_argument('--max_n', type=int, default=100, help='Maximum number of nodes in each component.')
parser.add_argument('--p', type=float, default=0.05, help='Probability of connection of Erdos-Renyi Model.')
parser.add_argument('--number_of_components', type=int, nargs='+', default=[1, 2, 3],
                    help='Number of connected components.')
parser.add_argument('--number_of_graphs', type=int, default=5000, help='Number of graphs per class.')
parser.add_argument('--artificially_connect', dest='artificially_connect', action='store_true',
                    help='Connect components using the original methodology of Dai et al. (2018).')
parser.add_argument('--no-artificially_connect', dest='artificially_connect', action='store_false',
                    help='Keep sampling components until a connected one is sampled.')
parser.set_defaults(artificially_connect=False)
parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
args = parser.parse_args()
setseed(args.seed)

def erdos_renyi_graph(min_n: int, max_n: int, p: float, connected_components: int, artificially_connect: bool) \
        -> dgl.DGLHeteroGraph:
    """ A graph with `connected_components` connected components. Each component is built by generating an ER(n, p)
    where min_n <= n <= max_n. Each component is then connected. The way the component is connected depends on
    the value of `artificially_connect`.

    Args:
        min_n: Minimum number of nodes in each connected component
        max_n: Maximum number of nodes in each connected component
        p: probability to connect nodes when generating an ER graph
        connected_components: number of connected components in the final graph
        artificially_connect: If true components are connected by adding an edge between successive connected
            components. If false new samples will be generated until a connected one is found.
    """
    components = [erdos_renyi_component(min_n, max_n, p, artificially_connect) for _ in range(connected_components)]
    graph = nx.disjoint_union_all(components)
    graph = dgl.from_networkx(graph)
    return graph


def erdos_renyi_component(min_n: int, max_n: int, p: float, artificially_connect: bool) -> nx.classes.graph.Graph:
    """Generates a single connected component."""
    n = np.random.randint(min_n, max_n+1)
    graph = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(graph):
        if artificially_connect:
            graph = artificially_connect_graph(graph)
        else:
            graph = nx.erdos_renyi_graph(n, p)
    return graph


def artificially_connect_graph(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
    """Connects a graph using the methodology of Dai et al. 2018."""
    edges = []
    all_components = list(nx.connected_components(graph))
    for i in range(len(all_components)-1):
        u = np.random.choice(list(all_components[i]))
        v = np.random.choice(list(all_components[i+1]))
        edges.append((u, v))
    graph.add_edges_from(edges)
    return graph


# generate data
total_number_of_samples = len(args.number_of_components) * args.number_of_graphs  # number of samples in dataset
dataset = []
with tqdm.tqdm(total=total_number_of_samples) as progress_bar:
    for label in args.number_of_components:
        for _ in range(args.number_of_graphs):
            graph = erdos_renyi_graph(args.min_n, args.max_n, args.p, label, args.artificially_connect)
            dataset.append((graph, torch.tensor([label])))
            progress_bar.update(1)

# save file
pickle.dump(dataset, open('erdos_renyi.pl', 'wb'))
