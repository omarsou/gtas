"""
Taken and adapted from https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction/sman.
"""

from tqdm import tqdm
import numpy as np
import dgl
import torch
import os
import pickle


class GenerateGraphData:
    def __init__(self,
                 data_path,
                 dataset_name,
                 data_flag,
                 dist_dim=4):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.data_flag = data_flag
        self.dist_dim = dist_dim

        self.dim_node_features = None
        self.dim_edge_features = None

        self.graph_list = []
        self.pk_list = []
        self._load_data()

    def _encode_dist_np(self, dist):
        dist = np.clip(dist, 1.0, 4.999)
        dist = dist - 1
        interval = 4.0 / self.dist_dim
        dist = dist / interval
        dist = dist.astype('int32')
        dist_v = (np.arange(self.dist_dim) == dist[:, None]).astype(np.float32)
        return dist_v

    def _distance_uv(self, rela_pos):
        '''
        u_pos: [None, 3]
        v_pos: [None, 3]
        '''
        dist = np.sqrt(np.square(rela_pos).sum(axis=-1))
        dist_v = self._encode_dist_np(dist)
        # dist_v = np.array([self._encode_dist_4(d) for d in dist]).astype(np.float32)
        return dist, dist_v

    def _spatial_edge_feat(self, edges, coords):
        '''
        calculate spatial distance
        '''
        # distance
        edges = np.array(edges)
        u_pos, v_pos = coords[edges[:, 0]], coords[edges[:, 1]]
        rela_pos = u_pos - v_pos  # edge feat: relative position vector with shape (None, 3)
        dist_scalar, dist_feat = self._distance_uv(rela_pos)

        return dist_feat, dist_scalar

    def _load_data(self):
        """Loads dataset
        """
        filename = os.path.join(self.data_path,
                                "{0}_{1}.pickle".format(self.dataset_name, self.data_flag))
        print("loading data from {}".format(filename))

        with open(filename, 'rb') as reader:
            data_drugs, data_Y = pickle.load(reader, encoding='iso-8859-1')

        self.num_graph = len(data_drugs)
        self.pk_list = data_Y

        for d_graph in tqdm(data_drugs):
            num_nodes, features, edges, coords = d_graph  # int, ndarray, list, ndarray

            features = features.astype(np.float32)
            edges = [(i, j) for i, j in edges]

            # One Hot-encoding the distances

            dist_feat_onehot, dist_edges = self._spatial_edge_feat(edges, coords)

            # Now, creating a DGL graph out of d_graph :

            g = dgl.graph(data=edges, num_nodes=num_nodes)
            # Instantiate features for nodes (atoms)
            g.ndata["feat"] = torch.Tensor(features)
            # Instantiate features for edges (bound)
            ## Distance as scalar
            # g.edata['dist'] = torch.from_numpy(dist_edges)
            ## concatenation : node i, node j, onehotencode dist(i,j)
            features_edges = np.array(
                [np.concatenate([features[i], features[j], dist_feat_onehot[idx]]) for idx, (i, j) in enumerate(edges)])
            g.edata['feat'] = torch.from_numpy(features_edges)

            # Update list of graph
            self.graph_list.append(g)

        self.dim_node_features = features.shape[1]
        self.dim_edge_features = features_edges.shape[1]
