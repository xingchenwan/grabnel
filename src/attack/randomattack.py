"""Random attack."""
from copy import deepcopy

import dgl
import numpy as np
import pandas as pd
import torch

from .base_attack import BaseAttack
from .utils import correct_predictions, random_sample_rewire_swap, random_sample_flip, population_graphs, extrapolate_breakeven


class RandomFlip(BaseAttack):

    def __init__(self, classifier: torch.nn.Module, loss_fn: torch.nn.Module, mode: str = 'flip',
                 target_class: int = None, preserve_disconnected_components=False,
                 **kwargs):
        """A baseline attack that chooses pairs of nodes to flip edges between.
        mode: flip or rewire"""
        super().__init__(classifier, loss_fn)
        assert mode in ['flip', 'add', 'remove', 'rewire'], f'mode {mode} is not recognised!'
        self.mode = mode
        self.target_class = target_class
        self.preserve_disconnected_components = preserve_disconnected_components

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int):
        """

        Args:
            graph: Unperturbed graph. This graph will be copied before perturbing.

        Returns:
            A perturbed version of the input graph.
        """
        adv_example = None
        best_losses_so_far = []
        merged_dfs = None
        is_edge_weighted = 'weight' in graph.edata.keys()
        for i in range(max_queries):
            if i % 100 == 0:
                print(f'Iter: {i} / {max_queries} = {i/max_queries*100} %. Best loss={np.max(merged_dfs.losses.values) if merged_dfs is not None else "NA"}')
            # sample edges
            if self.mode == 'rewire':
                edges = random_sample_rewire_swap(graph, budget, rewire_only=not is_edge_weighted,
                                                  preserve_disconnected_components=self.preserve_disconnected_components)
            else:  # flip, add or remove
                edges = random_sample_flip(graph, budget, add_edge_only=self.mode == 'add',
                                           remove_edge_only=self.mode == 'remove',
                                           preserve_disconnected_components=self.preserve_disconnected_components)

            with torch.no_grad():
                perturbed_graph = population_graphs(graph, [edges], mode=self.mode)[0]

                predictions = self.classifier(perturbed_graph).detach()
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                losses = self.loss_fn(predictions, label, reduction='none').numpy()
                new_df = self.construct_dataframe(losses, predictions, label.squeeze(), i + 1)

                if merged_dfs is None:  merged_dfs = new_df
                else: merged_dfs = pd.concat([merged_dfs, new_df])
                best_losses_so_far.append(np.max(merged_dfs.losses.values))
                # predictions = self.classifier(dgl.batch(perturbed_graphs))
                # labels = torch.repeat_interleave(label, len(predictions))
                if (self.target_class is None and np.sum(
                        correct_predictions(predictions.numpy(), label.numpy())) < len(
                    predictions)) \
                        or (self.target_class is not None and (
                        np.argmax(predictions.numpy(), axis=1) == self.target_class).any()):
                    print(f'Attack succeeded!: recent loss={losses}')
                    if self.target_class is None:
                        comps = correct_predictions(predictions.numpy(), label.numpy())
                        for i, comp in enumerate(comps):
                            if not comp:
                                adv_example = deepcopy(perturbed_graph)
                    else:
                        for i, pred in enumerate(predictions):
                            if np.argmax(pred.numpy()) == self.target_class:
                                adv_example = deepcopy(perturbed_graph)
                    return merged_dfs, adv_example

                if len(best_losses_so_far) > 200 and extrapolate_breakeven(best_losses_so_far) > 1e5:
                    print(f'Predicted breakeven point is {extrapolate_breakeven(best_losses_so_far)} and run terminated')
                    return merged_dfs, adv_example

        return merged_dfs, adv_example

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
