import sys

sys.path.append('../')

import argparse
import os
import torch
from functools import partial

from src.attack.data import Data
from src.attack.bayesopt_attack import BayesOptAttack
from src.attack.utils import (classification_loss, get_dataset_split, get_device,
                              setseed, nettack_loss)
from src.models.utils import get_model_class
import numpy as np

import datetime
import pickle

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run BO attack on Image Classification datasets')

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('-m', '--method', type=str, default='bo')
parser.add_argument('--loss', type=str, default='nettack')
parser.add_argument('--mode', type=str, default='rewire', choices=['rewire', 'flip'])
parser.add_argument('--model', type=str, default='chebygin', choices=['chebygin'])
parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
parser.add_argument('--gpu', type=str, default=None, help='A gpu device number if available.')
parser.add_argument('--n_trials', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--n_init', type=int, default=5)
parser.add_argument('--budget', type=float, default=0.015, )
parser.add_argument('--query_per_perturb', type=int, default=40, )
parser.add_argument('--n_samples', type=int, default=100, help='number of samples to attack')
parser.add_argument('--save_path', type=str, default='../src/output/attack_logs/',
                    help='save path for the output logs and adversarial examples.')
parser.add_argument('--model_path', type=str, default='../src/output/models/',
                    help='path for the trained classifier (victim models)')
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--target_class', type=int, default=None)
parser.add_argument('--loss_threshold', type=float, default=-3)
parser.add_argument('--density_threshold', type=float, default=0.5)
parser.add_argument('--max_h', type=int, default=1)
parser.add_argument('--no_greedy', action='store_true')
parser.add_argument('--acq', type=str, default='mutation', choices=['mutation', 'random'])
parser.add_argument('--exp_name', type=str, default=None, help='')


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
data = Data(dataset_name=dataset, dataset_split=dataset_split, seed=seed)
if args.target_class is not None:
    if args.exp_name is not None:
        save_path = args.save_path + f'/{args.exp_name}_{model_name}_{dataset}_{args.method}_{seed}_target_{args.target_class}/'
    else:
        save_path = args.save_path + f'/{model_name}_{dataset}_{args.method}_{seed}_{time_string}_target_{args.target_class}/'
else:
    if args.exp_name is not None:
        save_path = args.save_path + f'/{args.exp_name}_{model_name}_{dataset}_{args.method}_{seed}_untargeted/'
    else:
        save_path = args.save_path + f'/{model_name}_{dataset}_{args.method}_{seed}_{time_string}_untargeted/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

options = vars(args)
print(options)
option_file = open(save_path + "/command.txt", "w+")
option_file.write(str(options))
option_file.close()


model_class = get_model_class(model_name)
model_path = os.path.join(args.model_path, "checkpoint_mnist-75sp_139255_epoch30_seed0000111.pth.tar")
state = torch.load(model_path, map_location='cpu')
state_args = state['args']
model = model_class(data.feature_dim, data.number_of_labels, filters=state_args.filters, K=state_args.filter_scale,
                    n_hidden=state_args.n_hidden, aggregation=state_args.aggregation, dropout=state_args.dropout,
                    readout=state_args.readout, pool=state_args.pool, pool_arch=state_args.pool_arch)
model.load_state_dict(state['state_dict'])
model.eval()

correct_indices = []
for i in range(len(data.dataset_c)):
    sample = data.dataset_c[i]
    graph, label = sample
    preds = model(graph)
    if preds.argmax() == label:
        correct_indices.append(i)

print(f' Correctly classified samples: {len(correct_indices)} / {len(data.dataset_c)}')

n_success = 0
dfs, adv_examples = [], []

for trial in range(n_trials):
    selected_indices = np.random.RandomState(args.seed).choice(len(data.dataset_c), min(len(data.dataset_c), args.n_samples), replace=False).tolist()

    print(f'Starting trial {trial}/{n_trials}')
    for i, sample_id in enumerate(selected_indices):
        print(f'Starting sample {i} (Dataset ID={sample_id}')
        if sample_id in correct_indices:
            graph, label = data.dataset_c[i]
            if args.target_class is not None and label == args.target_class:
                continue
            if args.mode == 'rewire': # each rewire edit = 2 x flip edit. divide by 2
                edit = max(np.round(n_perturb * graph.num_nodes() ** 2).astype(int) // 2, 1)
                queries_per_perturb = args.query_per_perturb * 2
            elif args.mode == 'flip':
                edit = max(np.round(n_perturb * graph.num_nodes() ** 2).astype(int), 1)
                queries_per_perturb = args.query_per_perturb
            elif args.mode == 'linf':   # continuous perturbation
                assert 0. < n_perturb < 1.
                edit = n_perturb

            if args.target_class is not None: nettack = partial(nettack_loss, target_class=args.target_class)
            else: nettack = nettack_loss

            if args.method == 'bo':
                attacker = BayesOptAttack(model, nettack, surrogate='bayeslinregress',
                                          surrogate_settings={'h': args.max_h, 'extractor_mode': 'continuous', 'node_attr': 'node_attr'},
                                          batch_size=args.batch_size,
                                          edit_per_stage=min(5, edit) if args.no_greedy else 1,
                                          target_class=args.target_class,
                                          mode=args.mode,
                                          acq_settings={'acq_optimiser': args.acq},
                                          verbose=True,
                                          n_init=args.n_init,
                                          terminate_after_n_fail=args.patience)
            else:
                attacker = BayesOptAttack(model, nettack, surrogate='null',
                                          surrogate_settings={'h': args.max_h, 'extractor_mode': 'continuous', 'node_attr': 'node_attr'},
                                          batch_size=args.batch_size,
                                          edit_per_stage=min(5, edit) if args.no_greedy else 1,
                                          target_class=args.target_class,
                                          mode=args.mode,
                                          acq_settings={'acq_optimiser': args.acq},
                                          verbose=True,
                                          n_init=min(edit * queries_per_perturb, int(2e4)),
                                          terminate_after_n_fail=args.patience)
            df, adv_example = attacker.attack(graph, label, edit, min(edit * queries_per_perturb, int(2e4)))

            if adv_example is not None:
                n_success += 1
            dfs.append(df)
            adv_examples.append(adv_example)
        else:
            adv_examples.append(None)
            dfs.append(None)
        to_save = {
            'dataframes': dfs,
            'adv_example': adv_examples,
        }
        pickle.dump(dfs, open(os.path.join(save_path, f'trial-{trial}.pickle'), 'wb'))
        pickle.dump(adv_examples, open(os.path.join(save_path, f'trial-{trial}-adv_example.pickle'), 'wb'))