"""
meshgraphnet.py
Meshgraphnet model
"""

import torch
from src.gnn.models.models_helpers import build_mlp_d
import torch.nn as nn
from torch_scatter import scatter_add

class Encoder(torch.nn.Module):
    """
    Encoder for node and edge features, made by 2 mlps (1 each).
    """

    def __init__(self, node_channels, edge_channels, latent_and_out_size, layer_norm_enable):
        super(Encoder, self).__init__()
        self.edge_encoder = build_mlp_d(in_size=edge_channels, hidden_size=latent_and_out_size, out_size=latent_and_out_size,
                                        num_layers=1,
                                        layer_norm=layer_norm_enable,
                                        use_sigmoid=False, use_softmax=False)
        self.node_encoder = build_mlp_d(node_channels, latent_and_out_size, latent_and_out_size,
                                        num_layers=1,
                                        layer_norm=layer_norm_enable,
                                        use_sigmoid=False, use_softmax=False)

    def forward(self, node_features, edge_features):
        node_embedding = self.node_encoder(node_features)
        edge_embedding = self.edge_encoder(edge_features)
        return node_embedding, edge_embedding


class GraphNetBlock(torch.nn.Module):
    """
    Graph network block for message passing.

    Performs one round of message passing:
    1. Update edge embeddings based on sender and receiver nodes
    2. Aggregate messages at each node
    3. Update node embeddings
    """

    def __init__(self, in_size_edge_MLP, in_size_node_MLP, latent_and_out_size, num_layers_graphnetblock, enable_layer_norm):
        super(GraphNetBlock, self).__init__()
        self._latent_size = latent_and_out_size
        # Edge MLP: updates edge embeddings
        self.edge_net = build_mlp_d(in_size=in_size_edge_MLP, hidden_size=latent_and_out_size, out_size=latent_and_out_size,
                                    num_layers=num_layers_graphnetblock,
                                    layer_norm=enable_layer_norm,
                                    use_sigmoid=False, use_softmax=False)
        # Node MLP: updates node embeddings
        self.node_net = build_mlp_d(in_size=in_size_node_MLP, hidden_size=latent_and_out_size, out_size=latent_and_out_size,
                                    num_layers=num_layers_graphnetblock,
                                    layer_norm=enable_layer_norm,
                                    use_sigmoid=False, use_softmax=False)

    def forward(self, edge_index, node_embedding, edge_embedding):
        """
        Args:
            edge_index:

            node_embedding:
                node embedding (gave by the encoder)
            edge_embedding:
                node embedding (gave by the encoder)

        :return: (new_node_embedding, new_edge_embedding)
            forward pass of node and edge embedding
        """
        senders, receivers = edge_index
        num_nodes = node_embedding.shape[0]

        # Update edge embeddings: pass both node_embeddings
        edge_attr_ = self.edge_net(torch.hstack((node_embedding[senders], node_embedding[receivers], edge_embedding)))
        # For each node, aggregate its received messages FIXME: understand scatter_add
        # For each node i, sum all edge_attr_[e] such that receivers[e] == i.
        # agg = zeros([num_nodes, latent])
        # for e in range(E):
        #     agg[receivers[e]] += edge_attr_[e]
        # FIXME: why it receives agg_received_edges
        agg_received_edges = scatter_add(edge_attr_, receivers.long(), dim=0, dim_size=num_nodes)
        # Update node embeddings
        node_attr_ = self.node_net(torch.hstack((agg_received_edges, node_embedding)))
        # Residual connections
        new_node_embedding = node_attr_ + node_embedding
        new_edge_embedding = edge_attr_ + edge_embedding

        return new_node_embedding, new_edge_embedding


class Decoder(torch.nn.Module):
    """
    Final decoder to predict acceleration and von Mises stress.
    """

    def __init__(self, in_and_latent_size, enable_layer_norm):
        super(Decoder, self).__init__()
        self.acc_decoder = build_mlp_d(in_size=in_and_latent_size, hidden_size=in_and_latent_size, out_size=3,
                                       num_layers=1,
                                       layer_norm=enable_layer_norm,
                                       use_sigmoid=False, use_softmax=False)
        self.von_mises_decoder = build_mlp_d(in_size=in_and_latent_size, hidden_size=in_and_latent_size, out_size=1,
                                             num_layers=1,
                                             layer_norm=enable_layer_norm,
                                             use_sigmoid=False, use_softmax=False)

    def forward(self, node_embedding):
        """
        Given a node embedding, it returns the prediction for acceleration and for von mises stress.
        Doesn't use edge embedding.

        Args:
            node_embedding: torch.Tensor
                embedding of nodes
        :return: pred
        """
        decoded_acc = self.acc_decoder(node_embedding)
        decoded_von_mis = self.von_mises_decoder(node_embedding)
        pred = torch.hstack((decoded_acc, decoded_von_mis))
        return pred


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet Model.
    Made by:
    1) 1 encoder (2 mlps, one for edge feat one for node feat)
    2) Multiple message passing layers
    3) One final decoder/head (2 mlps, one for acceleration with dim_out=3, one for von mises stress (scalar))
    Processes mesh data through encoding, message passing, and decoding to predict node accelerations and stress.
    """

    def __init__(self, node_channels, edge_channels, encoder_latent_size, decoder_latent_size, processor_latent_size,
                 num_msgs, num_layers_graphnetblock, layer_norm_enable_encoder, layer_norm_enable_decoder,
                 layer_norm_enable_processor):
        """
        Args:
            node_channels: int
            edge_channels: int
            encoder_latent_size: int
            num_msgs: int
        """

        super(MeshGraphNet, self).__init__()

        # Encoder
        self.encoder = Encoder(node_channels=node_channels, edge_channels=edge_channels,
                               latent_and_out_size=encoder_latent_size, layer_norm_enable=layer_norm_enable_encoder)
        # Message-passing layers
        self.processer_list = nn.ModuleList(
            [GraphNetBlock(in_size_edge_MLP=3 * decoder_latent_size, in_size_node_MLP=2 * decoder_latent_size,
                           latent_and_out_size=decoder_latent_size, num_layers_graphnetblock=num_layers_graphnetblock,
                           enable_layer_norm=layer_norm_enable_processor)
             for _ in range(num_msgs)]
        )
        # Decoder
        self.decoder = Decoder(in_and_latent_size=processor_latent_size, enable_layer_norm=layer_norm_enable_decoder)

    def forward(self, ingraph):
        """

        Args:
            ingraph: input graph object PyG
        """
        node_features = ingraph.x
        edge_features = ingraph.edge_attr
        edge_index = ingraph.edge_index
        # Encode (embedding)
        node_embedding, edge_embedding = self.encoder(node_features, edge_features)
        # Process through message-passing layers
        for processor in self.processer_list:
            node_embedding, edge_embedding = processor(edge_index, node_embedding, edge_embedding)
        # Decode (prediction head)
        pred = self.decoder(node_embedding)

        return pred