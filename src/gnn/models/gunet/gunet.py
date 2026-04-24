"""
gunet.py

Parameters:
- in_channels (int) – Size of each input sample.
- hidden_channels (int) – Size of each hidden sample.
- out_channels (int) – Size of each output sample.
- depth (int) – The depth of the U-Net architecture.
- pool_ratios (float or [float], optional) – Graph pooling ratio for each depth. (default: 0.5)
- sum_res (bool, optional) – If set to False, will use concatenation for integration of skip connections instead
    summation. (default: True)
- act (torch.nn.functional, optional) – The nonlinearity to use. (default: torch.nn.functional.relu)
"""

import torch
from torch.nn import ReLU
from torch_geometric.nn.models.graph_unet import GraphUNet
from src.gnn.models.models_helpers import build_mlp_d

class GUNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, layer_norm, depth, pool_ratios, sum_res,
                 enable_embedder, encoder_layers, decoder_layers):
        """

        Args:
            in_channels:
            hidden_channels:
            out_channels:
            layer_norm:
            depth:
            pool_ratios:
            sum_res:
            enable_embedder:
            encoder_layers:
            decoder_layers:
        """
        super().__init__()
        self.enable_embedder = enable_embedder
        if self.enable_embedder:
            # [N, C_in] --> [N, H]
            self.encoder = build_mlp_d(in_size=in_channels, hidden_size=hidden_channels, out_size=hidden_channels,
                                       num_layers=encoder_layers, layer_norm=layer_norm, use_sigmoid=True,
                                       use_softmax=False)
            # NOte: Graph U-Net already has an initial GCN layer. [N, H] --> [N, H]
            self.graph_unet = GraphUNet(hidden_channels, hidden_channels, hidden_channels, depth, pool_ratios,
                                        sum_res, ReLU)
        else:
            # [N, C_in] --> [N, H]
            self.graph_unet = GraphUNet(in_channels, hidden_channels, hidden_channels, depth, pool_ratios,
                                        sum_res, ReLU)
        # Note: Graph U-Net already has a final decoder, here we append a head [N, H] --> [N, H]
        self.acc_head = build_mlp_d(in_size=hidden_channels, hidden_size=hidden_channels, out_size=3,
                                    num_layers=decoder_layers, layer_norm=layer_norm, use_sigmoid=True,
                                    use_softmax=False)
        self.mises_head = build_mlp_d(in_size=hidden_channels, hidden_size=hidden_channels, out_size=1,
                                    num_layers=decoder_layers, layer_norm=layer_norm, use_sigmoid=True,
                                    use_softmax=False)

    def forward(self, x):
        if self.enable_embedder:
            x = self.encoder(x)
        x = self.graph_unet(x)
        acc_pred = self.acc_head(x)
        mises_pred = self.mises_head(x)
        pred = torch.cat([acc_pred, mises_pred], dim=1)
        return pred