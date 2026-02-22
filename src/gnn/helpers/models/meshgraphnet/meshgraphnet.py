import torch
from src.gnn.helpers.models.models_helpers import build_mlp_d
import torch.nn as nn

class Encoder(torch.nn.Module):
    """Encoder for node and edge features."""

    def __init__(self, node_channels, edge_channels, latent_size=128):
        super(Encoder, self).__init__()
        self.edge_encoder = build_mlp_d(
            edge_channels, latent_size, latent_size, num_layers=1, lay_norm=True
        )
        self.node_encoder = build_mlp_d(
            node_channels, latent_size, latent_size, num_layers=1, lay_norm=True
        )

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

    def __init__(
            self,
            in_size_edge_MLP,
            in_size_node_MLP,
            latent_size=128,
    ):
        super(GraphNetBlock, self).__init__()
        self._latent_size = latent_size

        # Edge MLP: updates edge embeddings
        self.edge_net = build_mlp_d(
            in_size_edge_MLP, latent_size, latent_size, lay_norm=True
        )

        # Node MLP: updates node embeddings
        self.node_net = build_mlp_d(
            in_size_node_MLP, latent_size, latent_size, lay_norm=True
        )

    def forward(self, edge_index, node_embedding, edge_embedding):
        senders, receivers = edge_index
        num_nodes = node_embedding.shape[0]

        # Update edge embeddings
        edge_attr_ = self.edge_net(
            torch.hstack(
                (node_embedding[senders], node_embedding[receivers], edge_embedding)
            )
        )

        # Aggregate received messages at each node
        agg_received_edges = scatter_add(
            edge_attr_, receivers.long(), dim=0, dim_size=num_nodes
        )

        # Update node embeddings
        node_attr_ = self.node_net(torch.hstack((agg_received_edges, node_embedding)))

        # Residual connections
        new_node_embedding = node_attr_ + node_embedding
        new_edge_embedding = edge_attr_ + edge_embedding

        return new_node_embedding, new_edge_embedding


class Decoder(torch.nn.Module):
    """Decoder to predict acceleration and von Mises stress."""

    def __init__(self, latent_size=128):
        super(Decoder, self).__init__()
        self.acc_decoder = build_mlp_d(latent_size, latent_size, 3, lay_norm=False)
        self.von_mises_decoder = build_mlp_d(
            latent_size, latent_size, 1, lay_norm=False
        )

    def forward(self, node_embedding):
        decoded_acc = self.acc_decoder(node_embedding)
        decoded_von_mis = self.von_mises_decoder(node_embedding)
        pred = torch.hstack((decoded_acc, decoded_von_mis))
        return pred


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet Model.

    Processes mesh data through encoding, message passing, and decoding to predict node accelerations and stress.
    """

    def __init__(
            self,
            node_channels: int,
            edge_channels: int,
            latent_size: int = 128,
            num_msgs=3,
    ):
        super(MeshGraphNet, self).__init__()

        # Encoder
        self.encoder = Encoder(
            node_channels=node_channels,
            edge_channels=edge_channels,
            latent_size=latent_size,
        )

        # Message-passing layers
        self.processer_list = nn.ModuleList(
            [
                GraphNetBlock(
                    latent_size=latent_size,
                    in_size_edge_MLP=3 * latent_size,
                    in_size_node_MLP=2 * latent_size,
                )
                for _ in range(num_msgs)
            ]
        )

        # Decoder
        self.decoder = Decoder(latent_size)

    def forward(self, ingraph):
        node_features = ingraph.x
        edge_features = ingraph.edge_attr
        edge_index = ingraph.edge_index

        # Encode
        node_embedding, edge_embedding = self.encoder(node_features, edge_features)

        # Process through message-passing layers
        for processor in self.processer_list:
            node_embedding, edge_embedding = processor(
                edge_index, node_embedding, edge_embedding
            )

        # Decode
        pred = self.decoder(node_embedding)

        return pred