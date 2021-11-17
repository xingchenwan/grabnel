import dgl
import torch
from tqdm import tqdm
from copy import deepcopy
from .base_attack import BaseAttack
import pandas as pd
import numpy as np
from itertools import product
from functools import lru_cache


class GradArgMax(BaseAttack):

    def __init__(self, classifier, loss_fn, mode='flip', **kwargs):
        super().__init__(classifier, loss_fn)
        self.mode = mode

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int, verbose=True):
        """Attack graph by flipping edge with maximum gradient (in absolute value)."""
        graph = dgl.transform.remove_self_loop(graph)

        # save original graph and create a fully connected graph with binary edge weights which represent membership.
        unperturbed_graph = deepcopy(graph)
        m = unperturbed_graph.number_of_edges()
        graph, edge_weights = self.prepare_input(graph)

        # fast access to edge_ids
        self.graph = graph
        self.edge_ids.cache_clear()

        # initialise variables
        flipped_edges = set()
        losses = []
        correct_prediction = []
        queries = []
        progress_bar = tqdm(range(budget), disable=not verbose)

        # edge id for self loops if they exist
        self_loops = self.has_self_loops(graph)
        if self_loops:
            self_loop_ids = graph.edge_ids(graph.nodes(), graph.nodes(), return_uv=True)

        # sequential attack
        for i in progress_bar:

            # forward and backward pass
            predictions = self.classifier(graph, edge_weights)
            label_prediction = torch.argmax(predictions)
            loss = self.loss_fn(predictions, label)
            loss.backward()

            # update loss/query information
            losses.append(loss.item())
            queries.append(i)

            # stop early if attack is a success
            if label_prediction.item() != label.item():
                correct_prediction.append(False)
                break
            else:
                correct_prediction.append(True)

            # So the argmax chooses the most negative gradient for edges that
            # exist in the original graph or the most positive gradient for non-existent edges.
            gradients = edge_weights.grad.detach()
            gradients[:m] = -1 * gradients[:m]

            # mask gradients for already flipped edges so they cant be selected
            for edge in flipped_edges:
                edge_ids = self.edge_ids(edge)
                gradients[edge_ids] = -np.inf

            # mask self loops
            if self_loops:
                gradients[self_loop_ids] = -np.inf

            # mask gradients based on mode
            if self.mode == 'flip':
                pass
            elif self.mode == 'add':
                gradients[:m] = -np.inf
            elif self.mode == 'remove':
                gradients[m:] = -np.inf
            else:
                raise NotImplementedError('Only supports flip, add, remove.')

            # select edge to be flipped based on which flip will increase loss the most
            edge_index = torch.argmax(gradients).item()
            u = graph.edges()[0][edge_index].item()
            v = graph.edges()[1][edge_index].item()
            flipped_edge = frozenset((u, v))
            flipped_edges.add(flipped_edge)

            # update edge weights
            edge_weights = edge_weights.detach()
            edge_ids = self.edge_ids(flipped_edge)
            edge_weights[edge_ids] = 1 - edge_weights[edge_ids]
            edge_weights = edge_weights.requires_grad_(True)

            # update tqdm progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4}', 'selected': (u, v)})

        # prepare output information
        df = pd.DataFrame({'losses': losses,
                           'correct_prediction': correct_prediction,
                           'queries': queries})

        # construct adversarial example if the attack succeeds
        if not correct_prediction[-1]:
            adv_example = self.construct_perturbed_graph(unperturbed_graph, flipped_edges)
        else:
            adv_example = None

        # print if attack succeeded
        if verbose:
            if not correct_prediction[-1]:
                print('Attack success')
            else:
                print('Attack fail')

        return df, adv_example

    @staticmethod
    def prepare_input(graph):
        """Make graph fully connected but with zero weight edges where they don't exist"""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        to_add_u = []
        to_add_v = []
        for u, v in product(range(n), range(n)):
            if u != v and not graph.has_edges_between(u, v):
                to_add_u.append(u)
                to_add_v.append(v)
        edge_weights = torch.hstack((torch.ones(m), torch.zeros(len(to_add_u))))
        edge_weights.requires_grad = True
        graph.add_edges(to_add_u, to_add_v)
        return graph, edge_weights

    def construct_perturbed_graph(self, graph, flipped_edges):
        """Takes the unperturbed graph and list of flipped edges and applys the perturbation."""
        to_add_u = []
        to_add_v = []
        to_delete = []
        for edge in flipped_edges:
            u, v = edge
            if graph.has_edges_between(u, v):
                edge_ids = self.edge_ids(edge)
                to_delete += list(edge_ids.numpy())
            else:
                to_add_u += [u, v]
                to_add_v += [v, u]

        graph.remove_edges(to_delete)
        graph.add_edges(to_add_u, to_add_v)
        return graph

    @lru_cache(maxsize=None)
    def edge_ids(self, edge: frozenset):
        """Edge ids for edges u ~ v."""
        u, v = edge
        _, _, edge_ids_uv = self.graph.edge_ids([u], [v], return_uv=True)
        _, _, edge_ids_vu = self.graph.edge_ids([v], [u], return_uv=True)
        return torch.hstack((edge_ids_uv, edge_ids_vu))

    @staticmethod
    def has_self_loops(graph):
        """Determine if the graph contains self loops"""
        u = graph.nodes()
        return graph.has_edges_between(u, u).all().item()
