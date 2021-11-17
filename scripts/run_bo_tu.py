import sys
sys.path.append('../')
sys.path.append('../pytorch_structure2vec/s2v_lib')     # if doing s2v attack on er graphs.

import argparse
import os
from os.path import join

import pandas as pd
import torch

from src.attack.data import Data, ERData
from src.attack.bayesopt_attack import BayesOptAttack
from src.attack.genetic import Genetic
from src.attack.randomattack import RandomFlip
from src.attack.grad_arg_max import GradArgMax
from src.attack.utils import (classification_loss, get_dataset_split, get_device, setseed, nettack_loss)
from src.models.utils import get_model_class
import numpy as np

import datetime
import pickle
import dgl

parser = argparse.ArgumentParser(description='Run BO attack on TU datasets')

parser.add_argument('--dataset', type=str, default='COLLAB')
parser.add_argument('-m', '--method', type=str, default='bo')
parser.add_argument('--loss', type=str, default='nettack')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin', 's2v'])
parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
parser.add_argument('--gpu', type=str, default=None, help='A gpu device number if available.')
parser.add_argument('--n_trials', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--n_init', type=int, default=5)
parser.add_argument('--budget', type=float, default=0.03, )
parser.add_argument('--budget_by', type=str, choices=['nnodes_sq', 'nedges'], default='nnodes_sq',
                    help='computing method of the budget. nnodes_sq: budgets will be computed by nnodes^2.')
parser.add_argument('-qpp', '--query_per_perturb', type=int, default=40, )
parser.add_argument('--n_samples', type=int, default=100, help='number of samples to attack')
parser.add_argument('--save_path', type=str, default='../src/output/attack_logs/',
                    help='save path for the output logs and adversarial examples.')
parser.add_argument('--model_path', type=str, default='../src/output/models/',
                    help='path for the trained classifier (victim models)')
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--max_h', type=int, default=0)
parser.add_argument('--mode', type=str, default='flip', choices=['flip', 'rewire'])
parser.add_argument('--constrain_n_hop', type=int, default=None)
parser.add_argument('--no_greedy', action='store_true')
parser.add_argument('--acq', type=str, default='mutation', choices=['mutation', 'random'])
parser.add_argument('--exp_name', type=str, default=None, help='')
parser.add_argument('-pdc', '--preserve_disconnected_components', action='store_true',
                    help='whether constrain the attacks such as the number of disconnected components in the graph '
                         'does not change. ')

args = parser.parse_args()
setseed(args.seed)

seed = args.seed
n_trials = args.n_trials
n_samples = args.n_samples
n_perturb = args.budget
model_name = args.model
dataset = args.dataset
dataset_split = get_dataset_split(dataset)

# Time string will be used as the directory name
time_string = datetime.datetime.now()
time_string = time_string.strftime('%Y%m%d_%H%M%S')
if dataset == 'er_graphs':
    data = ERData(seed=seed)
else:
    data = Data(dataset_name=dataset, dataset_split=dataset_split, seed=seed)
if args.exp_name is None:
    save_path = args.save_path + f'/{model_name}_{dataset}_{args.method}_{seed}/'
else:
    save_path = args.save_path + f'/{args.exp_name}_{model_name}_{dataset}_{args.method}_{seed}_{time_string}/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

options = vars(args)
print(options)
option_file = open(save_path + "/command.txt", "w+")
option_file.write(str(options))
option_file.close()

model_class = get_model_class(model_name)
model = model_class(data.feature_dim, data.number_of_labels)
model_path = join(args.model_path, f'{args.model}_{dataset}_{seed}.pt')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

evaluation_logs = pd.read_csv(join('../src/output', 'evaluation_logs', f'{args.model}_{dataset}_{seed}.csv'))
evaluation_logs = evaluation_logs.query('dataset == "c"')

correct_indices = []

dataset_c_loader = data.adversarial_dataloaders()[1]
all_graphs = []
all_labels = []
for i, (graphs, labels) in enumerate(dataset_c_loader):
    with torch.no_grad():
        graphs = dgl.unbatch(graphs)
        all_graphs += graphs
        all_labels += labels.numpy().tolist()

for i in range(len(all_labels)):
    sample = all_graphs[i]
    preds = model(sample).detach()
    label = torch.tensor(all_labels[i])
    if data.is_binary and preds.shape[1] > 1:
        preds = preds[:, :1]

    if (preds.shape[1] == 1 and (preds > 0) == label) or (preds.shape[1] > 1 and preds.argmax() == label):
        assert evaluation_logs.iloc[i]['correct_prediction']
    if evaluation_logs.iloc[i]['correct_prediction']:
        correct_indices.append(i)
        graph = sample
        preds = model(graph).detach()

print(f' Correctly classified samples: {len(correct_indices)}')

all_labels = torch.tensor(all_labels)
n_success = 0
dfs, adv_examples = [], []

for trial in range(n_trials):

    selected_indices = np.random.RandomState(args.seed).choice(len(data.dataset_c), min(len(data.dataset_c), args.n_samples), replace=False).tolist()
    is_successful = [0] * len(selected_indices)
    is_attacked = [0] * len(selected_indices)
    is_correct = [0] * len(selected_indices)
    n_edges = [0] * len(selected_indices)
    n_nodes = [0] * len(selected_indices)

    print(f'Starting trial {trial}/{n_trials}')

    for i, sample_id in enumerate(selected_indices):
        n_stagnation = 0
        best_loss = -np.inf
        # try:
        n_nodes[i] = int(all_graphs[sample_id].num_nodes())
        n_edges[i] = int(all_graphs[sample_id].num_edges() // 2)
        print(f'Starting sample {i} (Dataset ID={sample_id}. Nnodes={n_nodes[i]}, Nedges={n_edges[i]}')

        if sample_id in correct_indices:
            is_attacked[i] = 1
            is_correct[i] = 1
            graph, label = all_graphs[sample_id], all_labels[sample_id].reshape(1, 1)

            if n_perturb >= 1:
                edit = int(min(n_perturb, 1))
                queries_per_perturb = args.query_per_perturb
            else:       # if expressed as fraction
                if args.mode == 'rewire':  # each rewire edit = 2 x flip edit. divide by 2
                    queries_per_perturb = args.query_per_perturb * 2
                    if args.budget_by == 'nnodes_sq':
                        edit = 1 + min(int(2e4 / queries_per_perturb),
                                       np.round(n_perturb * graph.num_nodes() ** 2 // 2).astype(int) // 2)
                    else:
                        edit = 1 + min(int(2e4 / queries_per_perturb),
                                       np.round(n_perturb * graph.num_edges() // 2 // 2).astype(int) // 2)
                else:
                    queries_per_perturb = args.query_per_perturb
                    if args.budget_by == 'nnodes_sq':
                        edit = 1 + min(int(2e4 / queries_per_perturb), np.round(n_perturb * graph.num_nodes() ** 2).astype(int))
                    else:
                        edit = 1 + min(int(2e4 / queries_per_perturb), np.round(n_perturb * graph.num_edges() // 2).astype(int))

            if args.method == 'bo':
                attacker = BayesOptAttack(model, nettack_loss,
                                          surrogate_settings={'h': args.max_h, 'extractor_mode': 'continuous'},
                                          batch_size=args.batch_size,
                                          edit_per_stage=min(5, edit) if args.no_greedy else 1,
                                          acq_settings={'acq_optimiser': args.acq,'rand_frac': 0., },
                                          verbose=True, mode=args.mode,
                                          n_init=args.n_init,
                                          terminate_after_n_fail=args.patience,
                                          n_hop_constraint=args.constrain_n_hop,
                                          preserve_disconnected_components=args.preserve_disconnected_components,)
            elif args.method == 'ga':  attacker = Genetic(model, nettack_loss, population_size=100, mode=args.mode)
            elif args.method == 'rs':  attacker = RandomFlip(model, nettack_loss, args.mode, preserve_disconnected_components=args.preserve_disconnected_components,)
            elif args.method == 'grad': attacker = GradArgMax(model, nettack_loss, args.mode)
            else: raise ValueError(f'Unknown method: {args.method}.')
            df, adv_example = attacker.attack(graph, label, edit, edit * queries_per_perturb)


            if adv_example is not None:
                n_success += 1
                is_successful[i] = 1
            dfs.append(df)
            adv_examples.append(adv_example)
        else:
            adv_examples.append(None)
            dfs.append(None)

        stats = {
            'selected_samples': selected_indices,
            'is_successful': is_successful,
            'is_attacked': is_attacked,
            'is_correct': is_correct,
            'nnodes': n_nodes,
            'nedges': n_edges,
        }

        pickle.dump(dfs, open(os.path.join(save_path, f'trial-{trial}.pickle'), 'wb'))
        pickle.dump(adv_examples, open(os.path.join(save_path, f'trial-{trial}-adv_example.pickle'), 'wb'))
        pickle.dump(stats, open(os.path.join(save_path, f'trial-{trial}-stats.pickle'), 'wb'))
