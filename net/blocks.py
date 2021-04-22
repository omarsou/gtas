"""
Taken and adapted from https://github.com/graphdeeplearning/graphtransformer
"""

import torch.nn as nn
import torch
import dgl.function as fn


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))

        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        e_out = g.edata['e_out']

        return h_out, e_out


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, drop=0.2, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, y):
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.activation(y)
            y = self.dropout(y)
        y = self.FC_layers[self.L](y)
        return y