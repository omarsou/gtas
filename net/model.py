"""
Taken and adapted from https://github.com/graphdeeplearning/graphtransformer
"""

import torch.nn as nn
import dgl

from net.blocks import MLPReadout
from net.layer import GraphTransformerLayer


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_features = net_params['num_atom_features']
        num_edge_input_dim = net_params['num_edge_input_dim']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        mlp_dropout = net_params['mlp_dropout']
        n_layers = net_params['L']
        pos_enc_dim = net_params['pos_enc_dim']
        type_loss = net_params['type_loss']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']

        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(num_atom_features, hidden_dim)

        self.embedding_e = nn.Linear(num_edge_input_dim, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1, drop=mlp_dropout)  # 1 out dim since regression problem

        if type_loss == "MSE":
            self.func_loss = nn.MSELoss()
        elif type_loss == "MAE":
            self.func_loss = nn.L1Loss()

    def forward(self, g, h, e, h_lap_pos_enc):

        # input embedding
        # Node Embedding and Positional Encoding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc

        # Edge Embedding
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        return self.func_loss(scores.float(), targets.float())