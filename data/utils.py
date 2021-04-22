"""
Taken and adapted from https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction/sman.
"""

from scipy import sparse as sp
import numpy as np
import dgl
import torch
import os
import pickle


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    return g


def add_laplacian_pos_encoding(train_graphs, val_graphs, test_graphs, pos_enc_dim):
    # Graph positional encoding v/ Laplacian eigenvectors
    train_graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in train_graphs]
    val_graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in val_graphs]
    test_graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in test_graphs]
    return train_graph_lists, val_graph_lists, test_graph_lists


def collate(samples):
    # The input samples is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels


def random_split(dataset_size, split_ratio=0.08, seed=0):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx


def split_train_valid(data_path, dataset_name, flag, seed=0):
    train_filename = os.path.join(data_path, "{0}_train.pickle".format(dataset_name))
    train_filename_ = os.path.join(data_path, "{0}_train_{1}.pickle".format(dataset_name, flag))
    valid_filename = os.path.join(data_path, "{0}_valid_{1}.pickle".format(dataset_name, flag))
    if os.path.isfile(valid_filename):
        return

    with open(train_filename, 'rb') as reader:
        data_drugs, data_Y = pickle.load(reader, encoding='iso-8859-1')

    train_idxs, valid_idxs = random_split(len(data_Y), split_ratio=0.9, seed=seed)
    train_drugs = [data_drugs[i] for i in train_idxs]
    valid_drugs = [data_drugs[i] for i in valid_idxs]
    train_y = [data_Y[i] for i in train_idxs]
    valid_y = [data_Y[i] for i in valid_idxs]

    train_data = (train_drugs, train_y)
    valid_data = (valid_drugs, valid_y)
    with open(train_filename_, 'wb') as f:
        pickle.dump(train_data, f)
    with open(valid_filename, 'wb') as f:
        pickle.dump(valid_data, f)
