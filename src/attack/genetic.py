"""Genetic algorithm attack."""
from copy import deepcopy

import dgl
import numpy as np
import pandas as pd
import scipy
import torch

from .base_attack import BaseAttack
from .utils import correct_predictions, population_graphs, random_sample_flip, random_sample_rewire_swap, get_allowed_nodes_k_hop, extrapolate_breakeven


class Genetic(BaseAttack):

    def __init__(self, classifier, loss_fn, population_size: int = 100,
                 crossover_rate: float = 0.1, mutation_rate: float = 0.2,
                 target_class: int = None,
                 mode: str = 'flip'):
        """A genetic algorithm based attack.

        This class stores an unperturbed graph in dgl.DGLGraph format, but the perturbed samples are represented as set.
        Each sample is a set of tuples (u, v) where u < v which represents an undirected edge u ~ v. To realise this
        perturbation each of the edges in the set are flipped. The original graph will be referred to as `graph` and an
        element of the population `sample`.

        Args:
            classifier: see BaseAttack
            loss_fn: see BaseAttack
            population_size: The number of perturbed graph in the population at any one point.
            crossover_rate: `crossover_rate` x `population_size` of the samples will be crossed over in each step.
            mutation_rate: All samples are mutated, `mutation_rate` of the flipped edges will be mutated.
            :param mode: str: 'flip', 'add', 'remove' or 'rewire': allowed edit operations on the edges.
        """
        super().__init__(classifier, loss_fn)
        self.target_class = target_class
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        assert mode in ['flip', 'add', 'remove', 'rewire'], f'mode {mode} is not recognised!'
        self.mode = mode

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int,
               initial_population: list = None):
        """The attack proceeds by rounds of selection, crossover and mutation. The number of rounds is determined by
        the `population_size` and `max_queries`. The population is a list of sets. Each set represents a perturbation.
        The set is of edge pairs (u, v) where u < v.

        initial_population: list: if specified, use this list of samples as initial population. Otherwise we randomly
            sample from the graphs
        """
        adv_example = None
        is_edge_weighted = 'weight' in graph.edata.keys()
        if initial_population is not None:
            self.population += initial_population
            if len(self.population) < self.population_size:
                if self.mode == 'rewire':
                    self.population += [random_sample_rewire_swap(graph, budget, rewire_only=not is_edge_weighted)
                                        for _ in
                                        range(self.population_size - len(self.population))]
                else:
                    self.population += [random_sample_flip(graph, budget) for _ in
                                        range(self.population_size - len(self.population))]
        else:
            self.population = self.initial_population(graph, budget)
        rounds = max(1, np.round(max_queries / self.population_size).astype(np.int))
        merged_dfs = None
        best_losses_so_far = []
        for round_no in range(rounds):
            fitness, predictions = self.fitness_of_population(graph, label, self.population)
            fitness = np.nan_to_num(fitness, neginf=0., posinf=0.)
            print(f'Round{round_no}/{rounds}: {np.max(fitness)}')
            self.population = self.select_fittest(self.population, fitness)
            self.population = self.crossover_population(self.population, budget)
            self.population = self.mutate_population(graph, self.population)
            new_df = self.construct_dataframe(fitness, predictions, label.squeeze(), (round_no + 1) * self.population_size)
            if merged_dfs is None: merged_dfs = new_df
            else: merged_dfs = pd.concat([merged_dfs, new_df])
            # added by xingchen: terminate the run whenever the attack succeeds.
            labels = torch.repeat_interleave(label, len(predictions))
            if (self.target_class is None and np.sum(correct_predictions(predictions.numpy(), labels.numpy())) < len(
                    predictions)) \
                    or (self.target_class is not None and (np.argmax(predictions.numpy(), axis=1) == self.target_class).any()):
                print('Attack succeeded!')
                if self.target_class is None:
                    comps = correct_predictions(predictions.numpy(), labels.numpy())
                    for i, comp in enumerate(comps):
                        if not comp:
                            adv_example = population_graphs(graph, [self.population[i]], mode=self.mode)
                            break
                else:
                    for i, pred in enumerate(predictions):
                        if np.argmax(pred.numpy()) == self.target_class:
                            adv_example = population_graphs(graph, [self.population[i]], mode=self.mode)
                            break
                break

            best_losses_so_far.append(np.max(merged_dfs.losses.values))
            if len(best_losses_so_far) > 200 / self.population_size and extrapolate_breakeven(best_losses_so_far) > 1e5 / self.population_size:
                print(f'Predicted breakeven point is {extrapolate_breakeven(best_losses_so_far)} and run terminated')
                break

        return merged_dfs, adv_example

    def initial_population(self, graph: dgl.DGLGraph, budget: int) -> list:
        """Create an initial population using random flips to create perturbation."""
        is_edge_weighted = 'weight' in graph.edata.keys()
        if self.mode == 'rewire':
            population = [random_sample_rewire_swap(graph, budget, rewire_only=not is_edge_weighted) for _ in
                          range(self.population_size - len(self.population))]
        else:
            population = [random_sample_flip(graph, budget) for _ in range(self.population_size)]
        return population

    def fitness_of_population(self, graph: dgl.DGLGraph, label: torch.tensor, population: list) \
            -> (np.array, torch.tensor):
        """Evaluate the fitness of the population.

        Args:
            graph: The original unperturbed graph
            label: The label of the unperturbed graph
            population: A list of perturbed graphs

        Returns:
            fitness: A 1D numpy array where the ith element is the loss of element i in the population
            predictions: A torch array containing logits (1D if its a binary classification task, otherwise an (n x C)
            array where C is the number of classes.
        """
        perturbed_graphs = population_graphs(graph, population, self.mode)
        with torch.no_grad():
            try:
                predictions = self.classifier(dgl.batch(perturbed_graphs))
            except RuntimeError:
                # this is possibly a dgl bug seemingly related to this https://github.com/dmlc/dgl/issues/2310
                # dgl.unbatch() should exactly inverses dgl.batch(), but you might get RuntimeError by doing something
                # like dgl.unbatch(dgl.batch([graphs])).
                predictions = self.classifier(perturbed_graphs)

            labels = torch.repeat_interleave(label, len(perturbed_graphs))
            fitness = self.loss_fn(predictions, labels, reduction='none')
        fitness = fitness.detach().numpy()
        return fitness, predictions

    def select_fittest(self, population: list, fitness: np.array) -> list:
        """Takes half the fittest scores and then samples the other half using softmax weighting on the scores."""
        softmax_fitness = scipy.special.softmax(fitness)
        fittest_idx = np.argsort(-softmax_fitness)[:int(np.floor(self.population_size / 2))]
        random_idx = np.random.choice(np.arange(self.population_size), int(np.ceil(self.population_size / 2)),
                                      replace=True, p=softmax_fitness)
        all_idx = np.concatenate([fittest_idx, random_idx])
        population = [population[idx] for idx in all_idx]
        return population

    def crossover_population(self, population: list, budget: int) -> list:
        """Each sample is crossed over by probability `self.crossover_rate`."""
        for i, sample in enumerate(population):
            if self.crossover_rate < np.random.rand():
                population[i] = self.crossover(sample, population, budget)
        return population

    def crossover(self, sample: set, population: list, budget: int) -> set:
        """Cross over of the `sample` and one other random sample from the `population`. The crossover is done by taking
        the union of all flips of the two samples and then sampling `budget` of them to create a new sample."""
        other_sample = np.random.choice(range(self.population_size))
        other_sample = population[other_sample]
        all_flips = list(set(sample).union(set(other_sample)))
        new_sample = np.random.choice(range(len(all_flips)), budget, replace=False)
        new_sample = set([all_flips[i] for i in new_sample])
        return new_sample

    def mutate_population(self, graph: dgl.DGLGraph, population: list) -> list:
        """Mutate all samples in the population."""
        for idx in range(self.population_size):
            population[idx] = self.mutate_sample(graph, population[idx])
        return population

    def mutate_sample(self, graph: dgl.DGLGraph, sample: set, ) -> set:
        """ Mutate the edges in the sample with at a rate of `self.mutation_rate`.

        Args:
            graph: The original unperturbed graph
            sample: The perturbed graph represented as a set of edge flips

        Returns:
            A new perturbed graph (in set format) which is a mutation of `sample`.
        """
        is_edge_weighted = 'weight' in graph.edata.keys()
        new_sample = set()

        # choose edges to mutate
        to_mutate = []
        for i, edge in enumerate(sample):
            if np.random.rand() < self.mutation_rate:
                to_mutate.append(edge)
            else:
                new_sample.add(edge)

        # mutate edges for new sample
        for edge in to_mutate:
            new_edge = self.mutate_rewire_triplet(graph, edge, rewire_only=not is_edge_weighted) \
                if self.mode == 'rewire' \
                else self.mutate_edge(graph, edge, )
            while new_edge in new_sample:
                new_edge = self.mutate_rewire_triplet(graph, edge, rewire_only=not is_edge_weighted) \
                    if self.mode == 'rewire' \
                    else self.mutate_edge(graph, edge,)
            new_sample.add(new_edge)

        return new_sample

    @staticmethod
    def mutate_edge(graph, edge, ):
        """Mutate a single edge.  The mutation chooses a random end point of the edge and then pairs it with a random
        node in the graph.
        """
        u, v = edge
        if np.random.rand() < 0.5:
            new_u = u
        else:
            new_u = v
        available_nodes = np.arange(graph.number_of_nodes())

        new_v = np.random.choice(available_nodes)
        while new_u == new_v:
            new_v = np.random.choice(available_nodes)

        return min(new_u, new_v), max(new_u, new_v)

    @staticmethod
    def mutate_rewire_triplet(graph, edge, rewire_only: bool = False, swap_only: bool = False):
        """Mutate triplet (u, v, w) used for rewiring operation (i.e. we either rewire u->v to u->w, or for the case
        when (u, w) is already an edge, swap u-v and u-w"""
        from copy import deepcopy
        if rewire_only and swap_only: raise ValueError(
            'Only either or neither of swap_only and rewire_only can be True!')
        # the index of the triplet to mutate
        patience = 100
        new_edge = deepcopy(edge)
        u, v, w = new_edge

        while patience >= 0:
            rewire_id = np.random.randint(0, len(edge))
            if rewire_id == 0:  # the candidate u is the neighbours of v with index number  < v
                new_node = np.random.choice(graph.out_edges(v)[1])
                if new_node in [u, v, w] or new_node > v:
                    patience -= 1
                    continue
                new_edge = (new_node, v, w)
                break
            elif rewire_id == 1:  # the candidate v is the neighbour of u with index number > u
                new_node = np.random.choice(graph.out_edges(u)[1])
                if new_node in [u, v, w] or new_node < u:
                    patience -= 1
                    continue
                new_edge = (u, new_node, w)
                break
            elif rewire_id == 2:
                if swap_only:
                    new_node = np.random.choice(graph.out_edges(u)[1])
                    if new_node in [u, v, w] or new_node < u:
                        patience -= 1
                        continue
                else:
                    new_node = np.random.randint(u, graph.number_of_nodes())
                    if new_node in [u, v, w]:
                        patience -= 1
                        continue
                    if rewire_only and new_node in graph.out_edges(u)[1]:
                        patience -= 1
                        continue
                new_edge = (u, v, new_node)
                break
        if patience <= 0:
            # print(f'Patience exhausted in trying to mutate {edge}!')
            return new_edge
        return new_edge

    @staticmethod
    def construct_dataframe(losses: np.array, predictions: torch.tensor, label: torch.tensor, queries: int) \
            -> pd.DataFrame:
        """Construct a pandas dataframe consistent with the base class. This dataframe is for all samples evaluated
        after exactly `queries` queries."""
        labels = np.tile(label, len(predictions))
        df = pd.DataFrame({'losses': losses,
                           'correct_prediction': correct_predictions(predictions.numpy(), labels),
                           'queries': queries})
        return df
