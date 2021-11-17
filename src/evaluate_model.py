"""Code to evaluate a graph classifier model."""
import argparse
import os
from os.path import join

import pandas as pd
import torch

from attack.data import Data, ERData
from attack.utils import (classification_loss, correct_predictions,
                          get_dataset_split, get_device, setseed)
from models.utils import get_model_class

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin', 's2v'])
parser.add_argument('--gpu', type=str, default=None, help='A gpu device number if available.')
parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
args = parser.parse_args()
setseed(args.seed)
print(vars(args))

# use gpu if available else cpu
device = get_device(args.gpu)

# load data
dataset_split = get_dataset_split(args.dataset)
if args.dataset == 'er_graphs':
    data = ERData(seed=args.seed)
else:
    data = Data(dataset_name=args.dataset, dataset_split=dataset_split, seed=args.seed)

# load model
model_class = get_model_class(args.model)
model = model_class(data.feature_dim, data.number_of_labels)
model_path = join(f'output/models/{args.model}_{args.dataset}_{args.seed}.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# specify loss function
loss_fn = classification_loss(data.is_binary)

# dataframe constructor
def results_to_df(predictions, labels, loss, number_of_labels):
    """ Constructs a dataframe summarising prediction information

    Args:
        predictions: An (n x l) numpy array where n are the number of samples and l is the number of labels (or 1 if binary)
        labels: A 1D array of predictions for each graph
        loss: A 1D array of the loss for the prediction and label
        number_of_labels: Number of labels in the dataset

    Returns:
        A dataframe with columns [labels, loss, predictions, correct_predictions]. Predictions takes the form of many
        columns if the number_of_labels > 2. In this case we have columns predictions_0, ... predictions_{l-1}.
    """
    results_dict = {'labels': labels, 'loss': loss, 'correct_prediction': correct_predictions(predictions, labels)}
    if number_of_labels == 2:
        results_dict.update({'predictions': predictions.squeeze()})
    else:
        for class_label in range(number_of_labels):
            results_dict.update({f'predictions_{class_label}': predictions[:, class_label]})
    return pd.DataFrame(results_dict)


# datasets to evaluate model on
dataset_b_loader, dataset_c_loader = data.adversarial_dataloaders()
results = []

# compute statistics for dataset b
for i, (graphs, labels) in enumerate(dataset_b_loader):
    with torch.no_grad():
        graphs, labels = graphs.to(device), labels.to(device)
        predictions = model(graphs)
        # GIN models still give a bug here:
        if data.is_binary and predictions.shape[1] > 1:
            predictions = predictions[:, :1]

        loss = loss_fn(predictions, labels, reduction='none')
        df = results_to_df(predictions.cpu().numpy(), labels.cpu().numpy(), loss.cpu().numpy(), data.number_of_labels)
    df['dataset'] = 'b'
    results.append(df)

# compute statistics for dataset c
for i, (graphs, labels) in enumerate(dataset_c_loader):
    with torch.no_grad():
        graphs, labels = graphs.to(device), labels.to(device)
        predictions = model(graphs)
        # GIN models still give a bug here:
        if data.is_binary and predictions.shape[1] > 1:
            predictions = predictions[:, :1]

        loss = loss_fn(predictions, labels, reduction='none')
        df = results_to_df(predictions.cpu().numpy(), labels.cpu().numpy(), loss.cpu().numpy(), data.number_of_labels)
    df['dataset'] = 'c'
    results.append(df)

# test set accuracy and loss
results = pd.concat(results)
print('Test set accuracy (b)', 100*results.query('dataset=="b"')['correct_prediction'].mean())
print('Test set loss (b)', results.query('dataset=="b"')['loss'].mean())
print('Test set accuracy (c)', 100*results.query('dataset=="c"')['correct_prediction'].mean())
print('Test set loss (c)', results.query('dataset=="c"')['loss'].mean())

# save data
os.makedirs(join('output', 'evaluation_logs'), exist_ok=True)
results_path = os.path.join('output', 'evaluation_logs', f'{args.model}_{args.dataset}_{args.seed}.csv')
results.to_csv(results_path)
