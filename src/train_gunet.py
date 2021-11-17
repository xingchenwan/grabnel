import sys
sys.path.append('../')
sys.path.append('./src/models/gunet')

import argparse
import random
import time
import torch
import numpy as np
from src.models.gunet.network import GNet
from src.models.gunet.trainer import Trainer
from src.models.gunet.utils.data_loader import FileLoader
from src.models.gunet.config import get_parser, update_args_with_default

parser = get_parser()
parser.add_argument('--save_path', type=str, default='../src/output/models/')
parser.add_argument('--log_path', type=str, default='../src/output/training_logs/')
parser.add_argument('--split_save_path', type=str, default='../data/')
parser.add_argument('-seed', '--seed', type=int, default=1, help='seed')
parser.add_argument('-data', '--data', default='PROTEINS', help='data folder name')
parser.add_argument('--preamble_path', type=str, default='../data/gunet_data/')
args, _ = parser.parse_known_args()

# for known datasets, overwrite with the default hyperparameters setting provided by the GUNet authors
args = update_args_with_default(args)


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    G_data.pickle_data()
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data, save_path=args.save_path, log_path=args.log_path)
    trainer.train()


def main():
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data = FileLoader(args).load_data(args.preamble_path)
    print('load data using ------>', time.time()-start)
    if args.fold == 0:
        for fold_idx in range(10):
            print('start training ------> fold', fold_idx+1)
            app_run(args, G_data, fold_idx)
    else:
        print('start training ------> fold', args.fold)
        app_run(args, G_data, args.fold-1)


if __name__ == "__main__":
    main()
