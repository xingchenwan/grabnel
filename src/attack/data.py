"""
Using the convention of having an a, b, c dataset used in ReWatt.

Dataset a is used for training a model, the method training_dataloaders returns two dataloaders created by splitting
dataset a. The first dataloader is for training and the other for validation

Dataset b is used for training the adversarial attack agent.

Dataset c is used to evaluate the adversarial attack agent.

Dataset b and c are returned when calling advesarial_dataloaders.
"""

import os
import pickle
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import random
from dgl import add_self_loop
from dgl.data import MiniGCDataset, TUDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from scipy.sparse.csgraph import connected_components


class Data:

    def __init__(self, dataset_name='REDDIT-MULTI-5K', dataset=None, dataset_split=(0.9, 0.05, 0.05), valid_size=0.2,
                 seed=None, generator_specs: dict = {}):
        """
        A Dataclass which downloads, stores and handles splitting of a dataset.

        :param dataset_name: a TUDataset name (listed at https://chrsmrrs.github.io/datasets/docs/datasets/)
        :param dataset: a dataset in the format of [(G1, y1), (G2, y2)...]. If this is supplied, the values input here
            will be used as dataset and this overrides any dataset_name specification.
        :param dataset_split: relative size of dataset a, b and c (see docstring for what these are)
        :param valid_size: proportion of dataset a to assign to validation
        :param seed: the seed which determine the dataset splits
        :param generator_specs: dict. Used for generative datasets (e.g. the Mini graph classification dataset
        in DGL, which expects arguments such as the number of graphs, the min/max nodes and etc.

        changed the default data_split -- 0.02 is too small for some datasets.
        """
        assert np.isclose(np.sum(dataset_split), 1.0)
        self.dataset_name = dataset_name
        self.valid_size = valid_size
        self.dataset_split = dataset_split
        self.seed = seed

        if dataset is None and dataset_name is None:
            raise ValueError("Either dataset or dataset_name must be provided! but got None for both.")
        if dataset is None:
            self.dataset = get_dataset(dataset_name, **generator_specs)
        else:
            self.dataset = dataset
        dataset_a, dataset_b, dataset_c = self.three_way_split(self.dataset, self.dataset_split)
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.dataset_c = dataset_c
        dataset_a_train, dataset_a_valid = self.dataset_a_split()
        self.dataset_a_train = dataset_a_train
        self.dataset_a_valid = dataset_a_valid
        if dataset_name.lower() == 'twitter':
            self.impute_and_normalise()

        self.feature_dim = self.dataset[0][0].ndata['node_attr'].shape[1]
        self.number_of_labels = len(np.unique([datapoint[1] for datapoint in self.dataset]))
        self.is_binary = self.number_of_labels == 2
        self.generator_specs = generator_specs

    def three_way_split(self, dataset, dataset_split):
        """
        Splits a dataset of the form [(G1, y1), (G2, y2)...] into three stratified datasets of the same form.

        :param dataset: An iterable of items (Gi, yi) where Gi is a DGLGraph and yi is an int (the label)
        :param dataset_split: A tuple (a, b, c) such that a+b+c=1.0 describing size of each dataset
        :return: the three datasets
        """
        _, b, c = dataset_split
        graphs, labels = map(list, zip(*dataset))  # [(G1, y1), (G2, y2)...] -> [[G1, G2,...]. [y1, y2,...]]
        a_graphs, bc_graphs, a_labels, bc_labels = \
            train_test_split(graphs, labels, test_size=b + c, stratify=labels, random_state=self.seed)
        b_graphs, c_graphs, b_labels, c_labels = \
            train_test_split(bc_graphs, bc_labels, test_size=c / (b + c), stratify=bc_labels, random_state=self.seed)
        dataset_a = list(zip(a_graphs, a_labels))
        dataset_b = list(zip(b_graphs, b_labels))
        dataset_c = list(zip(c_graphs, c_labels))
        return dataset_a, dataset_b, dataset_c

    def dataset_a_split(self):
        """Split dataset_a into train and validation."""
        graphs_a, labels_a = map(list, zip(*self.dataset_a))  # [(G1, y1), (G2, y2)...] -> [[G1, G2,...]. [y1, y2,...]]
        train_graphs, valid_graphs, train_labels, valid_labels = \
            train_test_split(graphs_a, labels_a, test_size=self.valid_size, random_state=self.seed)
        train = list(zip(train_graphs, train_labels))  # [G1, G2,...]. [y1, y2,...] -> [(G1, y1), (G2, y2)...]
        valid = list(zip(valid_graphs, valid_labels))
        return train, valid

    def impute_and_normalise(self):
        """Impute and normalise datasets that are returned as dataloaders."""
        pipe = self.build_pipe()
        self.dataset_a_train = self.apply_pipe_to_dataset(pipe, self.dataset_a_train)
        self.dataset_a_valid = self.apply_pipe_to_dataset(pipe, self.dataset_a_valid)
        self.dataset_b = self.apply_pipe_to_dataset(pipe, self.dataset_b)
        self.dataset_c = self.apply_pipe_to_dataset(pipe, self.dataset_c)

    def build_pipe(self):
        """Build a pipe fitted to dataset_a_train."""
        graphs = list(zip(*self.dataset_a_train))[0]
        features = dgl.batch(graphs).ndata['node_attr'].numpy()
        pipe = Pipeline([('impute', SimpleImputer()), ('scale', StandardScaler())])
        pipe.fit(features)
        return pipe

    def apply_pipe_to_dataset(self, pipe, dataset):
        """Apply an sklearn PipeLine to the node_attr of a DGL.DGLGraphs in the dataset."""
        return [(self.apple_pipe_to_graph(pipe, graph), label) for (graph, label) in dataset]

    @staticmethod
    def apple_pipe_to_graph(pipe, graph):
        """Apply an sklearn PipeLine to the node_attr of a DGL.DGLGraph."""
        graph.ndata['node_attr'] = torch.FloatTensor(pipe.transform(graph.ndata['node_attr'].numpy()))
        return graph

    def training_dataloaders(self, batch_size=32):
        """
        Returns two dataloaders, one for training a model and one for validating a model. The dataloaders come from
        dataset a.

        :param valid_size: The proportion of dataset a going into the validation set
        :param batch_size: size of batches used in dataloaders
        :return: two dataloaders
        """
        train_loader = DataLoader(self.dataset_a_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_loader = DataLoader(self.dataset_a_valid, batch_size=batch_size, collate_fn=collate)
        return train_loader, valid_loader

    def adversarial_dataloaders(self, batch_size=32, shuffle_b=False):
        """
        Returns dataset b and c used for training and evaluating an adversarial attack agent.

        :param batch_size: size of batches used in dataloaders
        :return: wo dataloaders
        """
        train_loader = DataLoader(self.dataset_b, batch_size=batch_size, shuffle=shuffle_b, collate_fn=collate)
        valid_loader = DataLoader(self.dataset_c, batch_size=batch_size, collate_fn=collate)
        return train_loader, valid_loader


class ERData:

    def __init__(self, dataset_split=(0.9, 0.09, 0.01), seed=None, **kwargs):
        """ER graphs dataset used in rls2v"""
        self.dataset_split = dataset_split
        if seed is None:
            raise ValueError('Specify seed to match model seed.')
        data_location = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'data')

        # load datasets
        self.dataset_a = pickle.load(open(os.path.join(data_location, 'er_train_dgl.pickle'), 'rb'))
        self.dataset_bc = pickle.load(open(os.path.join(data_location, 'er_test_dgl.pickle'), 'rb'))

        # split in the same way as rls2v code
        random.seed(seed)
        random.shuffle(self.dataset_bc)
        proportion_b = dataset_split[1]/(dataset_split[1]+dataset_split[2])
        size_b = int(len(self.dataset_bc) * proportion_b)

        self.dataset_b = self.dataset_bc[:size_b]
        self.dataset_c = self.dataset_bc[size_b:]

        # set other attributes
        self.feature_dim = 1
        self.number_of_labels = 3
        self.is_binary = False

    def training_dataloaders(self, batch_size=32):
        train_loader = DataLoader(self.dataset_bc, batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_loader = DataLoader(self.dataset_c, batch_size=batch_size, collate_fn=collate)
        return train_loader, valid_loader

    def adversarial_dataloaders(self, batch_size=32, shuffle_b=False):
        train_loader = DataLoader(self.dataset_b, batch_size=batch_size, shuffle=shuffle_b, collate_fn=collate)
        valid_loader = DataLoader(self.dataset_c, batch_size=batch_size, collate_fn=collate)
        return train_loader, valid_loader


def get_dataset(dataset_name, **kwargs):
    """
    Returns an iterable where each item is of the (DGLGraph, int).
    The DGLGraph has node features in graph.ndata['node_attr']

    :param dataset_name: a name from TUDataset or one of 'er_graphs', 'minigc', 'Twitter'.
    :param specs: the specification arguments to be passed to the graph generator.
    :return: an iterable dataset
    """
    if dataset_name.lower() == 'er_graphs':
        dataset = get_er_dataset()
    # elif dataset_name.lower() == 'triangles':
    #     dataset = get_triangles()
    elif dataset_name.lower() == 'minigc':
        dataset = get_minigc_dataset(**kwargs)
    elif dataset_name.lower() == 'twitter':
        dataset = get_twitter_dataset()
    elif dataset_name.lower() == 'mnist':
        dataset = get_mnist75sp()
    else:
        dataset = get_tu_dataset(dataset_name)
    return dataset


def get_er_dataset():
    """
    Benchmark used in Dai 2018. To run load this dataset, you have to first run src/data/er_generator.py

    :return:A list of (DGLGraph, label)
    """
    dataset = pickle.load(open('data/erdos_renyi.pl', 'rb'))
    dataset = add_synthetic_features(dataset)
    dataset = [(graph, label-1) for (graph, label) in dataset]  # label is #connected_components-1 so labels start at 0
    return dataset


def get_tu_dataset(dataset_name):
    """
    Return a TUDataset in (DGLGraph, int) format. If the graph doesn't have 'node_attr' it will use node degree.

    :param dataset_name: the name of the TUDataset
    :return: A list of (graph, label)
    """
    dataset = TUDataset(dataset_name)
    dataset = add_synthetic_features(dataset)
    return dataset


def get_minigc_dataset(num_graphs=1000, min_num_v=80, max_num_v=100):
    """
    A wrapper for the following dataset: https://docs.dgl.ai/en/0.4.x/api/python/data.html#dgl.data.MiniGCDataset.
    """
    dataset = MiniGCDataset(num_graphs, min_num_v, max_num_v)
    dataset = add_synthetic_features(dataset)
    return dataset


def get_mnist75sp():
    data_location = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, 'data')
    dataset = pickle.load(open(os.path.join(data_location, 'mnist_75sp.p'), 'rb'))
    return dataset


def add_synthetic_features(dataset):
    """
    Adds a one hot encoded vector based on degree of the nodes.

    :param dataset: An iteratable of tuples (graph, label)
    :return: A list of (graph, label) where graph.ndata['node_attr'] is a pytorch feature matrix
    """
    # generate a encoding scheme
    observed_degrees = set()
    for graph, label in dataset:
        observed_degrees = observed_degrees.union(set(graph.in_degrees().numpy()))
    one_hot_encode_map = {degree: i for i, degree in enumerate(observed_degrees)}

    # generate features
    new_dataset = []
    for graph, label in dataset:
        encoding = torch.zeros((graph.num_nodes(), len(observed_degrees)))
        for i, node_degree in enumerate(graph.in_degrees().numpy()):
            encoding[i][one_hot_encode_map[node_degree]] = 1.
        graph.ndata['node_attr'] = encoding
        new_dataset.append((graph, label))
    return new_dataset


def get_twitter_dataset(balance=True, minimum_nodes=5):
    """
    Returns the Twitter dataset from https://science.sciencemag.org/. Many of the graphs are tiny (a large number are
    a single node). The dataset has three labels TRUE, FALSE and MIXED. The labels are imbalanced. To address these
    concerns the function arguments lets one filter the graph size and balance the dataset

    :param balance: If True the dataset is balanced by discarding samples from the majority class.
    :param minimum_nodes: Only return graphs with nodes at least minimum_nodes
    :return: A list of (DGLGraph, int) tuples of the form x_i, y_i
    """
    data_location = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'data')
    dataset = pickle.load(open(os.path.join(data_location, 'twitter.pl'), 'rb'))
    dataset = [sample for sample in dataset if sample[0].num_nodes() >= minimum_nodes]
    if balance:
        dataset = balance_indices(dataset)
    return dataset


def balance_indices(dataset):
    """Downsampling the majority classes."""
    labels = [sample[1] for sample in dataset]
    minority_label = pd.Series(labels).value_counts().argmin()
    minority_label_count = pd.Series(labels).value_counts().min()
    other_labels = list(set(np.unique(labels)) - set([minority_label]))
    indices = np.where(labels == minority_label)[0]
    for other_label in other_labels:
        indices = np.append(indices, np.random.choice(np.where(labels == np.int64(other_label))[0],
                                                      size=minority_label_count, replace=False))
    dataset = [sample for i, sample in enumerate(dataset) if i in indices]
    return dataset


def collate(samples, add_selfloops=True):
    """Used to create DGL dataloaders."""
    graphs, labels = map(list, zip(*samples))
    if add_selfloops:
        graphs = [add_self_loop(graph) for graph in graphs]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
