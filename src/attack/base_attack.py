import dgl
import pandas as pd
import torch


class BaseAttack:

    def __init__(self, classifier, loss_fn):
        """Base adversarial attack model

        Args:
            classifier: The pytorch classifier to attack.
            loss_fn: The loss function, this will be maximised by an attacker.
        """
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.results = None

    def attack(self, graph: dgl.DGLGraph, label: torch.tensor, budget: int, max_queries: int) -> pd.DataFrame:
        """

        Args:
            graph: The unperturbed input graph.
            label: The label of the unperturbed input graph.
            budget: Total number of edge additions and deletions.
            max_queries: The total number of times the victim model can be queried.

        Returns:
            A pandas dataframe with columns `losses`, `correct_prediction` and `queries`. As the attack progresses
            all intermediate perturbations with this information is stored. `losses` is the loss of the perturbed
            sample, `correct_prediction` is if the model still classifies the perturbed sample correctly and `queries`
            is the number of times the model has been queried when generating the sample.
        """
        pass
