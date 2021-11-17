"""Code to train a graph classifier model.
To be used to train any model except for the GraphUNet in the paper.
See train_gunet.py for the script to train Graph UNet.
"""
import argparse
import os
from copy import deepcopy
from os.path import join

import pandas as pd
import torch
import torch.optim as optim

from attack.data import Data, ERData
from attack.utils import (classification_loss, get_dataset_split, get_device,
                          number_of_correct_predictions, setseed)
from models.utils import get_model_class

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin', 'embedding', 's2v'])
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--gpu', type=str, default=None, help='A gpu device number if available.')
parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
args = parser.parse_args()
setseed(args.seed)
print(vars(args))

# use gpu if availbale else cpu
device = get_device(args.gpu)

# load data
dataset_split = get_dataset_split(args.dataset)
if args.dataset == 'er_graphs':
    data = ERData(seed=args.seed)
else:
    data = Data(dataset_name=args.dataset, dataset_split=dataset_split, seed=args.seed)

# specific model
model_class = get_model_class(args.model)
model = model_class(data.feature_dim, data.number_of_labels)
model = model.to(device)

# specify loss function
loss_fn = classification_loss(data.is_binary)

# train model
train_loader, valid_loader = data.training_dataloaders()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
best_val_acc = 0.
best_model = None
training_logs = []
for epoch in range(args.num_epochs):

    # training step
    model.train()
    train_loss, train_acc = 0, 0
    for i, (graphs, labels) in enumerate(train_loader):
        graphs, labels = graphs.to(device), labels.to(device)
        labels = labels.long()
        predictions = model(graphs)
        # GIN models still give a bug here:
        if data.is_binary and predictions.shape[1] > 1:
            predictions = predictions[:, 0]

        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
        train_acc += number_of_correct_predictions(predictions, labels, data.is_binary).detach().item()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)

    # evaluation step
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for i, (graphs, labels) in enumerate(valid_loader):
            graphs, labels = graphs.to(device), labels.to(device)
            labels = labels.long()
            predictions = model(graphs)
            if data.is_binary and predictions.shape[1] > 1:
                predictions = predictions[:, 0]

            loss = loss_fn(predictions, labels)
            valid_loss += loss.detach().item()
            valid_acc += number_of_correct_predictions(predictions, labels, data.is_binary).detach().item()
        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader.dataset)

    # save best model
    if valid_acc > best_val_acc:
        print('Best val acc recorded at epoch ', epoch)
        best_model = deepcopy(model)
        best_val_acc = valid_acc

    print(epoch, '{:.4f}'.format(train_loss), '{:.4f}'.format(valid_loss),
          '{:.2f}'.format(train_acc), '{:.2f}'.format(valid_acc))
    training_logs.append([epoch, train_loss, valid_loss, train_acc, valid_acc])

# save model
os.makedirs(join('output', 'models'), exist_ok=True)
model_path = join('output', 'models', f'{args.model}_{args.dataset}_{args.seed}.pt')
torch.save(best_model.state_dict(), model_path)

# save training information
os.makedirs(join('output', 'training_logs'), exist_ok=True)
training_logs_path = join('output', 'training_logs', f'{args.model}_{args.dataset}_{args.seed}.csv')
training_logs = pd.DataFrame(training_logs, columns=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
training_logs.to_csv(training_logs_path)
