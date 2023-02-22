import torch
from torch.nn import Linear, ModuleList, Module, Sequential
from torch_geometric.nn import GATConv
# from layers.gat_encoder import GraphAttentionEncoder
# from layers.graph_encoder import GraphAttentionEncoder
from .GAT import GraphAttentionEncoder
from .mlp import MLPdecoder
from torch_geometric.nn import global_mean_pool, global_max_pool

torch.manual_seed(42)

class GATnar(Module):
    def __init__(self, opts):
        super(GATnar, self).__init__()
        self.in_channels=opts.embedding_dim
        self.out_channels=opts.embedding_dim
        self.heads=opts.n_heads
        self.dropout=opts.dropout
        self.n_encode_layers=opts.n_encode_layers
        self.n_decode_layers=opts.n_decode_layers
        
        self.edge_dim = opts.edge_dim
        self.add_self_loops = False
        self.node_dim=4
        self.aggregation=opts.aggregation
        self.normalization=opts.normalization
        self.learn_norm=opts.learn_norm
        self.track_norm=opts.track_norm
        self.gated=opts.gated

        # self.num_layers=3

        self.init_embed = Linear(self.node_dim, self.in_channels)
        # self.encoder = GraphAttentionEncoder(n_layers=self.n_encode_layers, 
        #                         n_heads=self.heads,
        #                         hidden_dim=self.in_channels, 
        #                         aggregation=self.aggregation, 
        #                         norm=self.normalization, 
        #                         learn_norm=self.learn_norm,
        #                         track_norm=self.track_norm,
        #                         gated=self.gated
        #                         )
        self.encoder = GraphAttentionEncoder(n_layers=self.n_encode_layers, 
                                n_heads=self.heads,
                                hidden_dim=self.in_channels)

        # self.encoder = GATConv(in_channels=self.in_channels,
        #                         out_channels = self.in_channels,
        #                         heads=1, 
        #                         dropout=self.dropout,
        #                         edge_dim = self.edge_dim,
        #                         )
        
        # self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        # self.ReLU = torch.nn.ReLU()

        # self.encoder_layers = ModuleList(
        #     self.encoder for _ in range(self.n_encode_layers)
        # )

        self.decoder = MLPdecoder(hidden_dim=self.in_channels, 
                                    norm=self.normalization,
                                    learn_norm=self.learn_norm,
                                    track_norm=self.track_norm,
                                    n_layers=self.n_decode_layers)
        

    def forward(self, x, edge_index, edge_attr, batch_mask=None):
        # print(self.encoder_layers)
        x = self.init_embed(x)
        # for _ in range(self.n_encode_layers):
        #     for encode in self.encoder_layers:
        #         x = encode(x, edge_index, edge_attr)

        x = self.encoder(x, edge_index)
        # x = self.dropout(x)
        # x = self.ReLU(x)
        
        x = self.decoder(x, batch_mask)

        return x