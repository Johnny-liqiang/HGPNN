import os
import torch
import configparser
import argparse

from enum import IntEnum
from torch_geometric.data import Data

from MyBatch import Batch


def read_config(config_path):
    class MyConfigParser(configparser.ConfigParser):
        """
        set ConfigParser options for case sensitive.
        """
        def __init__(self, **kwargs):
            configparser.ConfigParser.__init__(self, **kwargs)

        def optionxform(self, optionstr):
            return optionstr
    parser = MyConfigParser(allow_no_value=True)
    parser.read(config_path)

    config_dict = {}
    for section in parser.sections():
        for option in parser.options(section):
            if parser.get(section, option) is None:
                config_dict[option] = None
            elif section == 'config_num':
                if option in ['lr', 'l2_decay', 'pooling_k', 'lin_dropout']:
                    config_dict[option] = parser.getfloat(section, option)
                else:
                    config_dict[option] = parser.getint(section, option)
            elif section == 'config_bool':
                config_dict[option] = parser.getboolean(section, option)
            else:
                config_dict[option] = parser.get(section, option)
    return config_dict


parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, help='config file', required=True)
args = parser.parse_args()
config_dict = read_config(args.c)

dataset_type = config_dict['dataset_type']
feature_type = config_dict['feature_type']

fold_root = config_dict['fold_root']

node_type_list = ['EEG', 'EOG', 'EMG', 'ECG']
num_type = len(node_type_list)
num_relation = num_type ** 2

lr = config_dict['lr']
epochs = config_dict['epochs']
batch_size = config_dict['batch_size']
k_fold = config_dict['k_fold']
n_HGTs = config_dict['n_HGTs']
n_SAGs = config_dict['n_SAGs']
n_heads = config_dict['n_heads']
emb_dim = config_dict['emb_dim']
lin_dim = config_dict['lin_dim']
lin2_dim = config_dict['lin2_dim']
pooling_k = config_dict['pooling_k']
lin_dropout = config_dict['lin_dropout']

readout_mode = config_dict['readout_mode']

shuffle = config_dict['shuffle']
self_loop = config_dict['self_loop']

output_root = config_dict['output_root']
criterion_root = os.path.join(output_root, 'criterion')
plot_root = os.path.join(output_root, 'plot')
adj_mat_root = config_dict['adj_mat_root']

folds_from = config_dict['folds_from'] - 1
folds_to = config_dict['folds_to']

device = torch.device('cuda')

if dataset_type == 'ISRUC-3':
    subject_set = set(range(10))
elif dataset_type == 'ISRUC-1':
    subject_set = set(range(100))
else:
    raise RuntimeError('Unexpected dataset name!')


class Electrode(IntEnum):
        EEG_F3 = 0
        EEG_C3 = 1
        EEG_O1 = 2
        EEG_F4 = 3
        EEG_C4 = 4
        EEG_O2 = 5
        EOG_Right = 6
        EOG_Left = 7
        EMG_Chin = 8
        ECG = 9
        EMG_Leg = 10


def get_edge_type(node1, node2):
    i = get_node_type(node1.item())
    j = get_node_type(node2.item())
    return i * 4 + j


def get_node_type(node):
    if Electrode(node).name[:3] == 'EEG':
        return 0
    elif Electrode(node).name[:3] == 'EOG':
        return 1
    elif Electrode(node).name[:3] == 'EMG':
        return 2
    elif Electrode(node).name[:3] == 'ECG':
        return 3


def get_batches(data, label, edge_index, fold):
    all_graphs = []
    batched_train_graphs = []
    batched_val_graphs = []

    for subject in range(len(subject_set)):
        node_type = []
        edge_type = []
        for i in range(data[subject].shape[1]):
            node_type.append([get_node_type(i)])
        for i, j in edge_index[subject].t():
            edge_type.append([get_edge_type(i, j)])
        node_type = torch.tensor(node_type, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        subject_graphs = []
        for i in range(data[subject].shape[0]):
            d = Data(x=data[subject][i], edge_index=edge_index[subject], y=label[subject][i].view(-1, 1))
            d.node_type = node_type
            d.edge_type = edge_type
            subject_graphs.append(d)
        all_graphs.append(subject_graphs)

    n_subject_in_fold = len(subject_set) // k_fold
    train_graphs = []
    for train_subjects in all_graphs[:fold * n_subject_in_fold] + all_graphs[(fold + 1) * n_subject_in_fold:]:
        for graph in train_subjects:
            train_graphs.append(graph)
    val_graphs = []
    for val_subjects in all_graphs[fold * n_subject_in_fold: (fold + 1) * n_subject_in_fold]:
        for graph in val_subjects:
            val_graphs.append(graph)

    i = 0
    while i * batch_size < len(train_graphs):
        print(f'train batch {i + 1}...')
        start = i * batch_size
        end = min(start + batch_size, len(train_graphs))
        train_batch = Batch.from_data_list(train_graphs[start: end])
        batched_train_graphs.append(train_batch)
        i += 1

    i = 0
    while i * batch_size < len(val_graphs):
        print(f'val batch {i + 1}...')
        start = i * batch_size
        end = min(start + batch_size, len(val_graphs))
        val_batch = Batch.from_data_list(val_graphs[start: end])
        batched_val_graphs.append(val_batch)
        i += 1

    return batched_train_graphs, batched_val_graphs
