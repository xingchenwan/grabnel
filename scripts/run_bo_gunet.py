import sys
sys.path.append('../')

from src.models.gunet.network import GNet, GUNet
import numpy as np
import pickle
import argparse
import os
import torch

from src.attack.genetic import Genetic
from src.attack.bayesopt_attack import BayesOptAttack
from src.attack.randomattack import RandomFlip
from src.attack.utils import (nettack_loss_gunet, setseed)
from src.models.gunet.utils.dataset import gunet_graph2dgl
from src.models.gunet.config import get_parser, update_args_with_default

# parser for the GUNet settings
parser = argparse.ArgumentParser(description='Args for graph prediction of GUNet classifier on TU datasets')
parser.add_argument('-m', '--method', type=str, default='bo')
parser.add_argument('--dataset', type=str, default='PROTEINS', choices=['COLLAB', 'IMDBMULTI', 'PROTEINS'])
# note: here we use the original designation by the GUNet authors
parser.add_argument('--seed', type=int, default=1, help='RNG seed.')
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
parser.add_argument('--split_info_path', type=str, default='../data/', help='path to load the hyperparameters and the'
                                                                            'test set split of the experiment.')
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--max_h', type=int, default=1)
parser.add_argument('--no_greedy', action='store_true')
parser.add_argument('--acq', type=str, default='mutation', choices=['mutation', 'random'])
parser.add_argument('--exp_name', type=str, default=None, help='')
parser.add_argument('--mode', type=str, default='flip')
parser.add_argument('--resume', action='store_true')

args, _ = parser.parse_known_args()
setseed(args.seed)

# -- 0. Set-up the saving paths, etc -- #
if args.exp_name is None or len(args.exp_name) == 0:
    save_path = args.save_path + f'/gunet_{args.dataset}_{args.method}_{args.seed}/'
else:
    save_path = args.save_path + f'/{args.exp_name}_gunet_{args.dataset}_{args.method}_{args.seed}_{args.time_string}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(f'Save path is {save_path}')

# -- 1. Load data splits and the hyperparameters of the victim model -- #
model_data = pickle.load(open(f'{args.split_info_path}/gunet_split_{args.dataset}_test_data_{args.seed}.pickle', 'rb'))
# get the default information of the victim models by loading the defaults provided by GUNet authors
config_parser = get_parser()
config_parser.add_argument('-seed', '--seed', type=int, default=args.seed, help='seed')
config_parser.add_argument('-data', '--data', default=args.dataset, help='data folder name')
configs = config_parser.parse_known_args()[0]
configs = update_args_with_default(configs)
model = GNet(model_data['feat_dim'], model_data['num_class'], configs)
# load the victim model and the test split generated during training of the victim model
model.load_state_dict(torch.load(f'{args.model_path}/gunet_{args.dataset}_{args.seed}.pt', map_location='cpu'))
test_data = model_data['test_split']
# initialise a classifier that can be used as victim model
m = GUNet(model, number_of_labels=model_data['num_class'], input_dim=model_data['feat_dim'])


# load and convert the graphs and labels into compatible data format
test_data_dgl = [gunet_graph2dgl(test_data[i]) for i in range(len(test_data))]
test_labels = torch.tensor([torch.tensor(g.label) for g in test_data]).reshape(-1, 1)

accs = m.is_correct(test_data_dgl, test_labels)
accs = accs.numpy().flatten()
print('Victim Model Accuracy:', np.sum(accs) / len(accs))

# statistics on the indices
correct_indices = np.argwhere(accs > 0).flatten()
is_successful = [0] * len(correct_indices)

dfs, adv_examples = [], []

correctly_classified_graphs = [test_data_dgl[i] for i in correct_indices]

n_nodes = [g.num_nodes() for g in correctly_classified_graphs]
n_edges = [g.num_edges() // 2 for g in correctly_classified_graphs]
offset = 0
if args.resume and os.path.exists(os.path.join(save_path, 'trial-0.pickle')):
    print('Existing records found. Resuming from past runs')
    try:
        dfs = pickle.load(open(os.path.join(save_path, 'trial-0.pickle'), 'rb'))
        adv_examples = pickle.load(open(os.path.join(save_path, 'trial-0-adv_example.pickle'), 'rb'))
        stats = pickle.load(open(os.path.join(save_path, 'trial-0-stats.pickle'), 'rb'))
        is_successful = stats['is_successful']
        n_nodes = stats['nnodes']
        n_edges = stats['nedges']
        correct_indices = stats['selected_samples'][len(dfs):]
        offset = len(dfs)
    except Exception as e:
        print(f'Loading failed with exception = {e}... restarting attacks from scratch')
        offset = 0

for i, selected_idx in enumerate(correct_indices):
    print(f'Starting sample {i+offset} (Dataset ID={selected_idx}. Nedges={n_edges[i+offset]}, Nnodes={n_nodes[i+offset]}')
    # selected_indices = correct_indices[0]
    graph, label = test_data_dgl[selected_idx], test_labels[selected_idx]
    if args.budget >= 1: edit = int(min(args.budget, 1))
    else:
        if args.budget_by == 'nnodes_sq':
            edit = 1 + min(int(2e4 / args.query_per_perturb), np.round(args.budget * graph.num_nodes() ** 2).astype(int))
        else:
            edit = 1 + min(int(2e4 / args.query_per_perturb), np.round(args.budget * graph.num_edges() / 2).astype(int))

    if args.method == 'bo':
        attacker = BayesOptAttack(m, nettack_loss_gunet, surrogate='bayeslinregress',
                                  surrogate_settings={'h': args.max_h, 'extractor_mode': 'continuous'},
                                  batch_size=args.batch_size,
                                  edit_per_stage=1,
                                  terminate_after_n_fail=args.patience,
                                  verbose=True,
                                  n_init=args.n_init, )
    elif args.method == 'seq_random':
        # note there we use the BO interface but the batch_size == n_init (all query points are randomly sampled!)
        attacker = BayesOptAttack(m, nettack_loss_gunet, surrogate='null',
                                  surrogate_settings={'h': args.max_h, 'extractor_mode': 'continuous'},
                                  batch_size=args.query_per_perturb,
                                  edit_per_stage=1,
                                  terminate_after_n_fail=args.patience,
                                  verbose=True,
                                  n_init=args.query_per_perturb, )
    elif args.method == 'random': attacker = RandomFlip(m, nettack_loss_gunet, args.mode)
    elif args.method == 'ga': attacker = Genetic(m, nettack_loss_gunet, population_size=50, )
    else: raise ValueError(f'Unknown method: {args.method}')

    df, adv_example = attacker.attack(graph, label, edit, edit * args.query_per_perturb)
    dfs.append(df)
    adv_examples.append(adv_example)
    if adv_example is not None:
        is_successful[i+offset] = 1

    stats = {
        'selected_samples': correct_indices,
        'is_successful': is_successful,
        'nnodes': n_nodes,
        'nedges': n_edges
    }

    pickle.dump(dfs, open(os.path.join(save_path, f'trial-0.pickle'), 'wb'))
    pickle.dump(adv_examples, open(os.path.join(save_path, f'trial-0-adv_example.pickle'), 'wb'))
    pickle.dump(stats, open(os.path.join(save_path, f'trial-0-stats.pickle'), 'wb'))
