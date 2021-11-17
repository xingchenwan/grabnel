from .base_attack import BaseAttack
from .genetic import Genetic
import torch
import dgl
import pandas as pd
import numpy as np
from bayesopt.bayesopt.predictors import GPWL, BayesianLinearRegression, NullSurrogate
from .utils import correct_predictions, random_sample_flip, random_sample_rewire_swap, population_graphs, extrapolate_breakeven
from copy import deepcopy


class BayesOptAttack(BaseAttack):
    def __init__(self, classifier: torch.nn.Module, loss_fn: torch.nn.Module,
                 batch_size: int = 1, n_init: int = 10,
                 edit_per_stage=None,
                 surrogate: str = 'bayeslinregress',
                 mode: str = 'flip',
                 target_class: int = None,
                 surrogate_settings: dict = None,
                 acq_settings: dict = None,
                 verbose: bool = True,
                 terminate_after_n_fail: int = None,
                 n_hop_constraint: int = None,
                 preserve_disconnected_components: bool = False,):
        """
        Attacking classifier via Bayesian optimisation with GP/Bayesian Linear regression surrogate with Weisfeiler-
            Lehman kernels.
        :param classifier: see BaseAttack
        :param loss_fn: see BaseAttack
        :param batch_size: the number of possible adversarial samples to propose at each BO iteration. Larger batch
            size will lead to faster performance, but correspondingly the performance might decrease
        :param edit_per_stage: int or float. the number of edits amortised to each stage. A smaller edit_per_stage leads
            to more stages which is more greedy, a larger edit_per_stage is less greedy but leads to a larger search
            space.
        :param surrogate: the choice of surrogate.
        :param n_init: the number of initial perturbations to be sampled randomly from the search space
        :param mode: str: 'flip', 'add', 'remove' or 'rewire': allowed edit operations on the edges.
        :param surrogate_settings: dict: any parameters to be passed to the surrogates. See bayesopt/gp_predictor.py
        :param acq_settings: dict: any parameters to be passed to the acquisition function.
        :param verbose: whether to enable diagnostic information.
        :param terminate_after_n_fail: the tolerance when the BO agent fails to push the attack loss. If this is not None (
            a positive int), after this number of successive failures in increasing attack loss the attack will be aborted.
        :param n_hop_constraint: int. If not None (a positive int), and edge perturbation (either rewire or flip) must
            be constrained within the n_hop distance of the first node.

        """
        super().__init__(classifier, loss_fn)
        self.target_class = target_class
        if acq_settings is None:
            acq_settings = {}
        if 'acq_type' not in acq_settings.keys(): acq_settings['acq_type'] = 'ei'
        if 'acq_optimiser' not in acq_settings.keys(): acq_settings['acq_optimiser'] = 'mutation'
        if 'acq_max_step' not in acq_settings.keys(): acq_settings['acq_max_step'] = 400
        if 'random_frac' not in acq_settings.keys(): acq_settings['random_frac'] = 0.5
        self.acq_settings = acq_settings
        self.batch_size = batch_size
        self.n_init = n_init
        self.edit_per_stage = edit_per_stage
        if surrogate_settings is None:
            surrogate_settings = {}

        if surrogate == 'gpwl':  self.surrogate = GPWL(**surrogate_settings)
        elif surrogate == 'bayeslinregress': self.surrogate = BayesianLinearRegression(**surrogate_settings)
        elif surrogate == 'null': self.surrogate = NullSurrogate()
        else: raise ValueError(f'Unrecognised surrogate choice {surrogate}')

        self.verbose = verbose
        assert mode in ['flip', 'add', 'remove', 'rewire'], f'mode {mode} is not recognised!'
        self.mode = mode
        # save a record of previous query history
        self.query_history = []
        self.loss_history = []
        self.terminate_after_n_fail = terminate_after_n_fail if terminate_after_n_fail is not None and terminate_after_n_fail > 0 else None
        self.n_hop_constraint = n_hop_constraint if n_hop_constraint is not None and n_hop_constraint > 0 else None
        self.preserve_disconnected_components = preserve_disconnected_components

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget, max_queries: int):
        """
        The main attack loop.
        - For BO, at each iteration, we only modify one edge. If we have budget > 1, we use a greedy approach to
            partition the total max_queries into int(max_queries/budget) stages. At each stage, we attack on the
            *base graph*: in the first stage, it is the original graph (i.e. graph passed as an argument here);
            in the subsequent stages, it is the best perturbed graph of the previous stage that led to the largest
            classifier loss.
        - The optimisation terminates once it detects a successful attack.

        For the rest, see documentation for Genetic and BaseAttack
        """
        if isinstance(budget, float):
            assert 0 < budget < 1., f'if a float is supplied, this number must be within 0 and 1 but got {budget}'
            budget = np.round(budget * graph.num_edges()).astype(np.int)
        if isinstance(self.edit_per_stage, float):
            self.edit_per_stage = np.round(self.edit_per_stage * graph.num_edges()).astype(np.int)
        stages, edits_per_stage = self.get_stage_statistics(max_queries, budget)
        if self.verbose:
            print(f'Total number of {max_queries} of queries is divided into {stages}')
            print(f'Edits per stage is {edits_per_stage}')
        self.query_history = []
        self.loss_history = []
        dfs = []
        self.committed_edits = []
        base_graph = graph
        i = 0
        adv_example = None
        is_edge_weighted = 'weight' in graph.edata.keys()

        best_loss = -np.inf
        n_fail = 0
        while i < max_queries:
            curr_stage = np.digitize(i, stages) - 1
            prev_stage = np.digitize(max(0, i - self.batch_size), stages) - 1
            edit_allowed_this_stage = edits_per_stage[curr_stage]
            if curr_stage != prev_stage or i == 0:
                if i > 0:
                    best_idx = torch.argmax(self.surrogate.y)
                    base_graph = deepcopy(self.surrogate.X[best_idx])
                    # update the list of prohibited edges
                    if len(self.query_history) > 0:
                        self.committed_edits += self.query_history[-self.surrogate.y.shape[0] + int(best_idx)]

                if self.verbose:
                    print(f'Entering Stage {curr_stage}. ')
                    print(f'Committed edge edits={self.committed_edits}')
                # sample randomly at the start of each stage
                n_init = min(self.n_init, stages[curr_stage + 1] - stages[curr_stage])
                if self.mode == 'rewire':
                    samples = [random_sample_rewire_swap(base_graph, edit_allowed_this_stage, rewire_only=not is_edge_weighted, n_hop=self.n_hop_constraint,
                                                         preserve_disconnected_components=self.preserve_disconnected_components
                                                         ) for _
                               in range(n_init)]
                else:
                    samples = [
                        random_sample_flip(base_graph, edit_allowed_this_stage, remove_edge_only=self.mode == 'remove',
                                           add_edge_only=self.mode == 'add', n_hop=self.n_hop_constraint,
                                           committed_edges=self.committed_edits,
                                           preserve_disconnected_components=self.preserve_disconnected_components,)
                        for _ in range(n_init)]
                if not len(samples):
                    print('Patience reached. Terminating the current run')
                    break

                perturbed_graphs = population_graphs(base_graph, samples, self.mode)
                self.query_history += samples
                i += n_init
            else:
                perturbed_graphs = self.suggest(base_graph, edit_allowed_this_stage, )
                i += self.batch_size

            with torch.no_grad():
                try:
                    preds = self.classifier(dgl.batch(perturbed_graphs))
                except:
                    preds = torch.cat([self.classifier(g) for g in perturbed_graphs])
                    if preds.ndimension() == 1:
                        preds.reshape(-1, 1)

                # dgl.batch and dgl.unbatch create lots of problems. use this as a fallback option
                # see reference in github issue:
                # https://github.com/dmlc/dgl/issues/2409
                if preds.shape[0] != len(perturbed_graphs):
                    preds = self.classifier(perturbed_graphs)

                if len(perturbed_graphs) == 1 and preds.shape[1] == 1:
                    labels = label[0].reshape(1)
                else:
                    labels = torch.repeat_interleave(label, len(perturbed_graphs))
                losses = self.loss_fn(preds, labels, reduction='none')
                if losses.ndimension() == 0:
                    losses = losses.reshape(1)
                self.loss_history += losses.detach().numpy().tolist()

                if self.verbose:
                    print(f'Iteration {i}. Loss: {losses.detach().numpy()}.')

            dfs.append(self.construct_dataframe(losses, preds, label.squeeze(), i + 1))

            if len(self.loss_history) > 200 and extrapolate_breakeven(self.loss_history) > 1e5:
                print(f'Predicted breakeven point {extrapolate_breakeven(self.loss_history)} and run terminated')
                break

            if (self.target_class is None and np.sum(correct_predictions(preds.numpy(), labels.numpy())) < len(perturbed_graphs)) \
                    or (self.target_class is not None and (np.argmax(preds.numpy(), axis=1) == self.target_class).any()):
                print('Attack succeeded!')
                if self.target_class is None:
                    comps = correct_predictions(preds.numpy(), labels.numpy())
                    for i, comp in enumerate(comps):
                        if not comp:
                            adv_example = perturbed_graphs[i]
                            break
                else:
                    for i, pred in enumerate(preds):
                        if np.argmax(pred.numpy()) == self.target_class:
                            adv_example = perturbed_graphs[i]
                            break
                break
            reset_surrogate = False
            self.observe(perturbed_graphs, losses, reset_surrogate=reset_surrogate)

            if np.max(losses.numpy()) > best_loss:
                n_fail = 0
                best_loss = torch.max(losses).detach().numpy()
            else:
                n_fail += len(perturbed_graphs)
            if self.terminate_after_n_fail is not None and n_fail > self.terminate_after_n_fail:
                print('Patience reached. Terminating the current run')
                break

        return pd.concat(dfs), adv_example

    def suggest(self, base_graph: dgl.DGLGraph, n_edit: int, prohibited_edges: list = None):
        """
        The BO function to suggest perturbations to be queried from self.classifier
        :param base_graph: the graph on which we perform perturbations
        :param n_edit: number of edge edit allowed per perturbation
        :param prohibited_edges: list of edge edits that are not allowed.
        :return: a list of dgl graphs of shape self.batch_size
        """
        is_edge_weighted = 'weight' in base_graph.edata.keys()
        candidate_graphs = None

        n_samples = self.acq_settings['acq_max_step']
        if self.acq_settings['acq_optimiser'] == 'random':
            if self.mode == 'rewire':
                candidate_samples = [random_sample_rewire_swap(base_graph,
                                                               n_edit,
                                                               rewire_only=not is_edge_weighted,
                                                               n_hop=self.n_hop_constraint,
                                                               preserve_disconnected_components=self.preserve_disconnected_components,
                                                               ) for _
                                     in range(n_samples)]
            else:
                candidate_samples = [random_sample_flip(base_graph, n_edit, remove_edge_only=self.mode == 'remove',
                                                        add_edge_only=self.mode == 'add',  n_hop=self.n_hop_constraint,
                                                        committed_edges=self.committed_edits,
                                                        preserve_disconnected_components=self.preserve_disconnected_components,

                                                        )
                                     for _ in range(n_samples)]

        elif self.acq_settings['acq_optimiser'] in ['genetic', 'mutation']:
            n_round = 10
            top_k = 3
            pop_size = max(n_samples // n_round, 100)
            # optionally set the fraction of randomly generated samples
            n_rand = np.round(pop_size * self.acq_settings['random_frac']).astype(np.int)
            n_mutate = pop_size - n_rand

            genetic_optimiser = Genetic(classifier=lambda x_: 0, loss_fn=lambda x_: 0,
                                        population_size=pop_size,
                                        mutation_rate=1., mode=self.mode)
            if self.mode == 'rewire':
                candidate_samples = [
                    random_sample_rewire_swap(base_graph, n_edit, rewire_only=not is_edge_weighted,
                                              n_hop=self.n_hop_constraint,
                                              preserve_disconnected_components=self.preserve_disconnected_components,

                                              ) for _ in
                    range(n_rand)]
            else:
                candidate_samples = [
                    random_sample_flip(base_graph, n_edit, remove_edge_only=self.mode == 'remove',
                                       add_edge_only=self.mode == 'add',
                                       n_hop=self.n_hop_constraint,
                                       committed_edges=self.committed_edits,
                                       preserve_disconnected_components=self.preserve_disconnected_components,

                                       ) for _ in range(n_rand)]

            self.query_history += candidate_samples
            topk_indices = torch.topk(self.surrogate.y, min(self.surrogate.y.shape[0], top_k))[1]
            while len(candidate_samples) < pop_size:
                selected_index = topk_indices[np.random.randint(len(topk_indices))]
                candidate_samples.append(
                    genetic_optimiser.mutate_sample(base_graph,
                                                    self.query_history[-len(self.surrogate.y) + selected_index],
                                                    )
                )
            candidate_graphs = population_graphs(base_graph, candidate_samples, self.mode)
            acq_values = self.surrogate.acquisition(candidate_graphs, acq_func=self.acq_settings['acq_type'], bias=None)

            # for each mutation round, alternate between optimising the topology (A) with features (X)
            for r in range(n_round):
                topk_indices = torch.topk(acq_values, min(len(candidate_graphs), top_k))[1]
                while len(candidate_samples) < pop_size:
                    selected_sample = candidate_samples[np.random.randint(len(topk_indices))]
                    candidate_samples.append(
                        genetic_optimiser.mutate_sample(base_graph, selected_sample,))
                candidate_samples = candidate_samples[n_mutate:]

                candidate_graphs = population_graphs(base_graph, candidate_samples, self.mode)
                acq_values = self.surrogate.acquisition(candidate_graphs, acq_func=self.acq_settings['acq_type'], bias=None)

        else:
            raise NotImplementedError(f'Unable to parse the acq_optimiser {self.acq_settings["acq_optimiser"]}')

        if candidate_graphs is None:
            candidate_graphs = population_graphs(base_graph, candidate_samples, self.mode)

        acq_values = self.surrogate.acquisition(candidate_graphs, acq_func=self.acq_settings['acq_type'])

        acq_values_np = acq_values.detach().numpy().flatten()
        acq_values_np_, unique_idx = np.unique(acq_values_np, return_index=True)
        i = np.argpartition(acq_values_np_, -min(acq_values_np_.shape[0], self.batch_size))[
            -min(acq_values_np_.shape[0], self.batch_size):]
        indices = np.array([unique_idx[j] for j in i])
        suggested = [candidate_graphs[j] for j in indices]
        self.query_history += [candidate_samples[j] for j in indices]
        return suggested

    def observe(self, X, y, reset_surrogate=False):
        """
        Update the BO with new sample-target pair(s) we obtained from quering the classifer
        :param X: a list of dgl graphs. The list of dgl graphs we queried from the classifier
        :param y: a Tensor of shape[0] = len(X). The tensor of the classifier loss
        :param reset_surrogate: whether to reset the surrogate (clearing all previous fitted (X, y)).
        """
        nan_idx = (y != y).nonzero().view(-1)
        if nan_idx.shape[0] > 0:
            for i in nan_idx:
                X.pop(i)
            y = y[y == y]
        if self.surrogate.X is None or reset_surrogate:
            self.surrogate.fit(X, y)
        else:
            self.surrogate.update(X, y)

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

    def get_stage_statistics(self, max_queries: int, budget: int):
        if self.edit_per_stage is None:
            self.edit_per_stage = budget
        if budget % self.edit_per_stage:
            num_stages = budget // self.edit_per_stage + 1
        else:
            num_stages = budget // self.edit_per_stage
        query_per_edit = max_queries // budget
        stage_length = self.edit_per_stage * query_per_edit
        stages = []
        edits_per_stages = []
        for i in range(num_stages):
            stages.append(min(max_queries, i * stage_length))
            if sum(edits_per_stages) + self.edit_per_stage < budget:
                edits_per_stages.append(self.edit_per_stage)
            else:
                edits_per_stages.append(budget - sum(edits_per_stages))
        stages.append(max_queries)
        return np.array(stages), np.array(edits_per_stages)
