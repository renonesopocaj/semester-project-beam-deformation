"""
meshgraphnet.py

MeshGraphNet model, encoder (embedding of nodes and edges) - processor (message passing layer updating both nodes and
edges/messages) - decoder (head for final prediction).
MLPs are ReLu activated.
"""

import torch
from src.gnn.models.models_helpers import build_mlp_d
import torch.nn as nn
from torch_scatter import scatter_add

class Encoder(torch.nn.Module):
    """
    Encoder for making an initial embedding of node and edge features, made by 2 mlps (1 MLP for each).
    """

    def __init__(self, node_channels, edge_channels, latent_and_out_size, layer_norm_enable, num_layers_encoder,
                 dropout):
        super(Encoder, self).__init__()
        # [N, C_n] --> [N, H]
        self.edge_encoder = build_mlp_d(in_size=edge_channels,
                                        hidden_size=latent_and_out_size,
                                        out_size=latent_and_out_size,
                                        num_layers=num_layers_encoder,
                                        layer_norm=layer_norm_enable,
                                        use_sigmoid=False, use_softmax=False,
                                        dropout=dropout)
        # [E, C_e] --> [E, H]
        self.node_encoder = build_mlp_d(in_size=node_channels,
                                        hidden_size=latent_and_out_size,
                                        out_size=latent_and_out_size,
                                        num_layers=num_layers_encoder,
                                        layer_norm=layer_norm_enable,
                                        use_sigmoid=False, use_softmax=False,
                                        dropout=dropout)

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

    def __init__(self, in_size_edge_MLP, in_size_node_MLP, latent_and_out_size, num_layers_graphnetblock,
                 enable_layer_norm, dropout):
        super(GraphNetBlock, self).__init__()
        self._latent_size = latent_and_out_size
        # Edge MLP: updates edge embeddings [IN, HIDD] --> [HIDD, HIDD]
        self.edge_net = build_mlp_d(in_size=in_size_edge_MLP,
                                    hidden_size=latent_and_out_size,
                                    out_size=latent_and_out_size,
                                    num_layers=num_layers_graphnetblock,
                                    layer_norm=enable_layer_norm,
                                    use_sigmoid=False, use_softmax=False,
                                    dropout=dropout)
        # Node MLP: updates node embeddings [IN, HIDD] --> [HIDD, HIDD]
        self.node_net = build_mlp_d(in_size=in_size_node_MLP, hidden_size=latent_and_out_size,
                                    out_size=latent_and_out_size,
                                    num_layers=num_layers_graphnetblock,
                                    layer_norm=enable_layer_norm,
                                    use_sigmoid=False, use_softmax=False,
                                    dropout=dropout)

    def forward(self, edge_index, node_embedding, edge_embedding):
        """
        Edge MLP in/out shapes: [E, 3H] --> [E, H]
        Node MLP in/out shapes: [N, 2H] --> [N, H]
        Args:
            edge_index: torch.Tensor [E, 2]
                matrix with edge indices (i,j) for each edge
            node_embedding: torch.Tensor [N, H]
                node embedding (gave by the encoder)
            edge_embedding: torch.Tensor [E, H]
                node embedding (gave by the encoder)

        :return: (new_node_embedding [E, H], new_edge_embedding [E, H])
            forward pass of node and edge embedding
        """
        senders, receivers = edge_index
        num_nodes = node_embedding.shape[0]
        # Update edge embeddings: pass both node_embeddings of i and j for (i,j) \in E
        # stacl([E, H], [E, H], [E, H]) = [E, 3H] --> [E, H]
        edge_attr_ = self.edge_net(torch.hstack((node_embedding[senders], node_embedding[receivers], edge_embedding)))
        # For each node i, aggregate its received messages i.e. sum all edge_attr_[e] such that receivers[e] == i.
        # agg = zeros([num_nodes, latent])
        # for e in range(E):
        #     agg[receivers[e]] += edge_attr_[e]
        agg_received_edges = scatter_add(edge_attr_, receivers.long(), dim=0, dim_size=num_nodes) # [N, H]
        # Update node embeddings stack([N, H] [, H]) = [N, 2H] --> [N, H]
        node_attr_ = self.node_net(torch.hstack((agg_received_edges, node_embedding)))
        # Residual connections
        new_node_embedding = node_attr_ + node_embedding
        new_edge_embedding = edge_attr_ + edge_embedding

        return new_node_embedding, new_edge_embedding


class Decoder(torch.nn.Module):
    """
    Final decoder to predict acceleration and von Mises stress. Uses 2 heads.
    """

    def __init__(self, in_and_latent_size, enable_layer_norm, num_layers_heads, dropout):
        super(Decoder, self).__init__()
        self.acc_decoder = build_mlp_d(in_size=in_and_latent_size, hidden_size=in_and_latent_size, out_size=3,
                                       num_layers=num_layers_heads,
                                       layer_norm=enable_layer_norm,
                                       use_sigmoid=False, use_softmax=False, dropout=dropout)
        self.von_mises_decoder = build_mlp_d(in_size=in_and_latent_size, hidden_size=in_and_latent_size, out_size=1,
                                             num_layers=num_layers_heads,
                                             layer_norm=enable_layer_norm,
                                             use_sigmoid=False, use_softmax=False, dropout=dropout)

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
    1) 1 encoder/embedder (2 mlps, one for edge feat one for node feat)
    2) Multiple message passing layers
    3) One final decoder/head (2 mlps, one for acceleration with dim_out=3, one for von mises stress (scalar))
    Processes mesh data through encoding, message passing, and decoding to predict node accelerations and stress.
    """

    def __init__(self, node_channels, edge_channels, encoder_latent_size, encoder_decoder_latent_size,
                 processor_latent_size, num_msgs, num_layers_graphnetblock, layer_norm_enable_encoder,
                 layer_norm_enable_decoder, layer_norm_enable_processor, num_layers_heads, num_layers_encoder,
                 dropout_encoder, dropout_head, dropout_processor):
        """
        Args:
            node_channels: int
                Specifies the size of the input feature vector for the nodes.
            edge_channels: int
                Specifies the size of the input feature vector for the edges.
            encoder_latent_size: int
                Defines the latent size in the encoder output.
            encoder_decoder_latent_size: int
                Latent size shared across both the encoder and decoder layers.
            processor_latent_size: int
                Latent size used in the processor layers.
            num_msgs: int
                Number of message-passing iterations in the graph processor.
            num_layers_graphnetblock: int
                Number of layers in each GraphNetBlock module in the processor.
            layer_norm_enable_encoder: bool
                Controls whether layer normalization is used in the encoder.
            layer_norm_enable_decoder: bool
                Controls whether layer normalization is used in the decoder.
            layer_norm_enable_processor: bool
                Controls whether layer normalization is used in the processor.
            num_layers_heads: int
                Number of layers in the decoder's head network for output prediction.
            num_layers_encoder: int
                Number of layers in the encoder's head network for node and edge embedding.
        """
        super(MeshGraphNet, self).__init__()
        # Encoder: takes nodes [N, C_n] --> [N, H] and edges [E, C_e] --> [E, H]
        self.encoder = Encoder(node_channels=node_channels, edge_channels=edge_channels,
                               latent_and_out_size=encoder_latent_size, layer_norm_enable=layer_norm_enable_encoder,
                               num_layers_encoder=num_layers_encoder, dropout=dropout_encoder)
        # L Message-passing layers: each one takes in input [N, 2H] --> [N, H] and edges [E, 3H] --> [E, H]
        self.processer_list = nn.ModuleList([GraphNetBlock(in_size_edge_MLP=3 * encoder_decoder_latent_size,
                                                           in_size_node_MLP=2 * encoder_decoder_latent_size,
                                                           latent_and_out_size=encoder_decoder_latent_size,
                                                           num_layers_graphnetblock=num_layers_graphnetblock,
                                                           enable_layer_norm=layer_norm_enable_processor,
                                                           dropout=dropout_processor)
                                             for _ in range(num_msgs)])
        # Decoder: takes in input [N, H] and outputs [N, 4] (acceleration and von Mises stress)
        self.decoder = Decoder(in_and_latent_size=processor_latent_size, enable_layer_norm=layer_norm_enable_decoder,
                               num_layers_heads=num_layers_heads, dropout=dropout_head)

    def forward(self, ingraph):
        """
        Args:
            ingraph: input graph object PyG
        """
        node_features = ingraph.x
        edge_features = ingraph.edge_attr
        edge_index = ingraph.edge_index
        # Encode (embedding). Nodes: [N, C_n] --> [N, H] | Edges: [E, C_e] --> [E, H]
        node_embedding, edge_embedding = self.encoder(node_features, edge_features)
        # Process through message-passing layers. Nodes: [N, 2H] --> [N, H] | Edges [E, 3H] --> [E, H]
        for processor in self.processer_list:
            node_embedding, edge_embedding = processor(edge_index, node_embedding, edge_embedding) # [N, H], [E, H]
        # Decode (prediction head)
        pred = self.decoder(node_embedding) # [N, 4]
        return pred