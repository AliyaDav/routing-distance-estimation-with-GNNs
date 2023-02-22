import torch
from torch.nn import Parameter, functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import GATConv
import numpy as np
from torch import nn
import math


def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=1, dim_size=num_nodes)[0][:, index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=1, dim_size=num_nodes)[:, index] + 1e-16)

    return out

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch', learn_norm=True, track_norm=False):
        super(Normalization, self).__init__()

        self.normalizer = {
            "layer": nn.LayerNorm(embed_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(embed_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(normalization, None)

    def forward(self, input, mask=None):
        if self.normalizer:
            return self.normalizer(
                input.view(-1, input.size(-1))
            ).view(*input.size())
        else:
            return input


class PositionWiseFeedforward(nn.Module):

    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)
    
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        # print('here')
        return input + self.module(input, mask)

class MultiHeadAttentionLayer(nn.Module):

    """Implements a configurable Transformer layer

    References:
        - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
        - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
    """

    def __init__(self, n_heads, embed_dim, feed_forward_dim, 
                 norm='batch', learn_norm=True, track_norm=False):
        super(MultiHeadAttentionLayer, self).__init__()

        self.self_attention = SkipConnection(
                GATConv(in_channels = embed_dim,
                        out_channels = embed_dim,
                        heads=n_heads, 
                        dropout=0.2,
                        concat=False,
                        # edge_dim = self.edge_dim,
                        )
            )
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                   embed_dim=embed_dim,
                   feed_forward_dim=feed_forward_dim
                )
            )
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    def forward(self, h, mask=None):
        h = self.self_attention(h, mask=mask)
        h = self.norm1(h, mask=mask)
        h = self.positionwise_ff(h, mask=mask)
        h = self.norm2(h, mask=mask)
        return h
    
class GraphAttentionEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch', 
                    learn_norm=True, track_norm=False, *args, **kwargs):
        super(GraphAttentionEncoder, self).__init__()
        
        feed_forward_hidden = hidden_dim * 4
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
                for _ in range(n_layers)
        ])

    def forward(self, x, edge_index=None, graph=None):
        for layer in self.layers:
            x = layer(x, edge_index)
            # print('shape of x in GraphAttentionEncoder', x.shape)
        return x