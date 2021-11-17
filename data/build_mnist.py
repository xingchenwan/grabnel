# generate the MNIST-75sp data in DGL format
from torchvision import datasets
import scipy.ndimage
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
import argparse
import numpy as np
import datetime
import os
import random
import pickle
import multiprocessing as mp
import networkx as nx
import dgl
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
parser.add_argument('-D', '--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('-o', '--out_dir', type=str, default='.', help='path where to save superpixels')
parser.add_argument('-s', '--split', type=str, default='train', choices=['train', 'val', 'test'])
parser.add_argument('-t', '--threads', type=int, default=0, help='number of parallel threads')
parser.add_argument('-n', '--n_sp', type=int, default=75, help='max number of superpixels per image')
parser.add_argument('-c', '--compactness', type=int, default=0.25, help='compactness of the SLIC algorithm '
                                                                        '(Balances color proximity and space proximity): '
                                                                        '0.25 is a good value for MNIST '
                                                                        'and 10 for color images like CIFAR-10')
parser.add_argument('--seed', type=int, default=111, help='seed for shuffling nodes')
parser.add_argument('--n_images', type=int, default=None, help='number of images to process. If none, process all MNIST'
                                                               'images.')
args = parser.parse_args()


def process_image(params):
    """Extract superpixels from the MNIST images using SLIC"""
    img, index, n_images, args, to_print, shuffle = params

    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    n_sp_extracted = args.n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    n_sp_query = args.n_sp + (
        20 if args.dataset == 'mnist' else 50)  # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    while n_sp_extracted > args.n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=args.compactness, multichannel=len(img.shape) > 2)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert args.n_sp >= n_sp_extracted > 0, (args.split, index, n_sp_extracted, args.n_sp)
    assert n_sp_extracted == np.max(superpixels) + 1, (
        'superpixel indices', np.unique(superpixels))  # make sure superpixel indices are numbers from 0 to n-1

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    # if to_print:
    #     print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index + 1, n_images, img.shape,
    #                                                                           img.min(), img.max(),
    #                                                                           sp_intensity.shape[0]))
    # Create edges btween nodes in the form of adjacency matrix
    sp_coord = sp_coord / img.shape[1]
    dist = cdist(sp_coord, sp_coord)
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2)
    A[np.diag_indices_from(A)] = 0

    mn = torch.tensor([0.11225057, 0.11225057, 0.11225057, 0.44206527, 0.43950436]).view(1, 1, -1)
    sd = torch.tensor([0.2721889, 0.2721889, 0.2721889, 0.2987583, 0.30080357]).view(1, 1, -1)

    node_features = ((torch.from_numpy(np.pad(np.concatenate((sp_intensity, sp_coord), axis=1),
                                              ((0, 0), (2, 0)), 'edge')).unsqueeze(0) - mn) / sd).numpy().squeeze()

    graph = build_graph(A, node_attributes={'node_attr': node_features}, graph_type='dgl')
    return graph, sp_intensity, sp_coord, sp_order, superpixels
    # return sp_intensity, sp_coord, sp_order, superpixels


def build_graph(adjacency_matrix: np.array, node_attributes: dict = None, graph_type='nx'):
    """
    Build a networkx or dgl graph from adjacency - node_attributes representation
    :param adjacency_matrix: numpy array
    :param node_attributes: optional. node attributes
    :param graph_type: 'nx' for networkx. every other string will yield dgl graph
    :return: graph (nx.Graph if graph_type == 'nx', otherwise a dgl.DGLGraph)
    """
    G = nx.from_numpy_array(adjacency_matrix)
    if node_attributes is not None:
        for n in G.nodes():
            for k, v in node_attributes.items():
                # print(k, v)
                G.nodes[n][k] = v[n]
    if graph_type == 'nx':
        return G
    G = G.to_directed()
    if node_attributes != None:
        node_attrs = list(node_attributes.keys())
    else:
        node_attrs = []
    g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
    return g


dt = datetime.datetime.now()
print('start time:', dt)

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

random.seed(args.seed)
np.random.seed(args.seed)  # to make node random permutation reproducible (not tested)

# Read image data using torchvision
is_train = args.split.lower() == 'train'
if args.dataset == 'mnist':
    data = datasets.MNIST(args.data_dir, train=is_train, download=True)
    assert args.compactness < 10, 'high compactness can result in bad superpixels on MNIST'
    assert 1 < args.n_sp < 28 * 28, 'the number of superpixels cannot exceed the total number of pixels or be too small'
elif args.dataset == 'cifar10':
    data = datasets.CIFAR10(args.data_dir, train=is_train, download=True)
    assert args.compactness > 1, 'low compactness can result in bad superpixels on CIFAR-10'
    assert 1 < args.n_sp < 32 * 32, 'the number of superpixels cannot exceed the total number of pixels or be too small'
else:
    raise NotImplementedError('unsupported dataset: ' + args.dataset)

images = data.train_data if is_train else data.test_data
labels = data.train_labels if is_train else data.test_labels
if not isinstance(images, np.ndarray):
    images = images.numpy()
if isinstance(labels, list):
    labels = np.array(labels)
if not isinstance(labels, np.ndarray):
    labels = labels.numpy()

n_images = args.n_images if args.n_images is not None else len(labels)
labels = labels[:n_images] if args.n_images is not None else labels

if args.threads <= 0:
    sp_data = []
    for i in tqdm(range(n_images)):
        sp_data.append(process_image((images[i], i, n_images, args, True, True)))
else:
    with mp.Pool(processes=args.threads) as pool:
        sp_data = pool.map(process_image, [(images[i], i, n_images, args, True, True) for i in range(n_images)])

superpixels = [sp_data[i][1:] for i in range(n_images)]
sp_data = [sp_data[i][0] for i in range(n_images)]

# structure the data into [(G1, y1), (G2, y2), ... ] format
save_data = list(zip(sp_data, torch.tensor(labels, dtype=torch.int32)))

with open('%s/%s_%dsp.p' % (args.out_dir, args.dataset, args.n_sp, ), 'wb') as f:
    pickle.dump(save_data, f, protocol=2)
with open('%s/%s_%dsp_superpixels.p' % (args.out_dir, args.dataset, args.n_sp, ), 'wb') as f:
    pickle.dump(superpixels, f, protocol=2)

print('done in {}'.format(datetime.datetime.now() - dt))
