import torch
from torch import nn

from gnn.graph_transformer import GraphEncoderLayer, LaplacianGraphEmbeddings


class GraphTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        node_features_size: int,
        number_eigenvectors: int,
        embedding_size: int,
        feedforward_scaling: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        bottleneck_size: int,
        device,
    ):
        super(GraphTransformerAutoencoder, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.bottleneck_size = bottleneck_size

        # Laplacian graph embeddings layer
        self.laplacian_embed_layer = LaplacianGraphEmbeddings(
            number_nodes,
            node_features_size,
            number_eigenvectors,
            embedding_size,
            device,
        )

        # Create the graph encoder (with all it's layers)
        self.graph_encoder = nn.ModuleList(
            [
                GraphEncoderLayer(
                    number_nodes,
                    embedding_size,
                    feedforward_scaling,
                    num_heads,
                    dropout,
                    self.device,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout1 = nn.Dropout(dropout)

        # Create bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_size, bottleneck_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.linear = nn.Linear(bottleneck_size, embedding_size)
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        self.graph_decoder = nn.ModuleList(
            [
                GraphEncoderLayer(
                    number_nodes,
                    embedding_size,
                    feedforward_scaling,
                    num_heads,
                    dropout,
                    self.device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graphs):
        # Compute graph embeddings
        node_embeddings, adjacency_matrices = self.laplacian_embed_layer(graphs)

        # Convert to tensor to use as residual connection and transpose for BatchNorm1d
        residual_connections_1 = node_embeddings.transpose(1, 2)

        encoded_outputs = node_embeddings
        for layer in self.graph_encoder:
            encoded_outputs = layer(
                encoded_outputs,
                adjacency_matrices,
                residual_connections_1,
            )

        # Dropout to combat overfitting
        encoded_outputs = self.dropout1(encoded_outputs)

        # Pass them to the bottleneck
        bottleneck = self.bottleneck(encoded_outputs)

        # Restore original dimensions
        decoder_inputs = self.linear(bottleneck)
        decoder_inputs = self.relu(decoder_inputs)
        decoder_inputs = self.dropout2(decoder_inputs)

        residual_connections_1 = decoder_inputs.transpose(1, 2)

        # Pass through decoder stack, final output is trained to be the same as encoder stack inputs
        decoder_outputs = decoder_inputs
        for layer in self.graph_encoder:
            decoder_outputs = layer(
                decoder_outputs, adjacency_matrices, residual_connections_1
            )
        bottleneck = bottleneck.reshape(len(graphs), -1)
        return bottleneck, decoder_outputs, node_embeddings


class ClassifierDNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        dropout: float,
        device,
    ):
        super(ClassifierDNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.dnn = nn.Sequential(
            nn.Linear(input_size, int(input_size * 3 / 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(input_size * 3 / 4), int(input_size * 2 / 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(input_size * 2 / 4), int(input_size * 1 / 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(input_size * 1 / 4), int(input_size * 1 / 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(input_size * 1 / 8), num_classes),
        )

    def forward(self, embeddings):
        return self.dnn(embeddings)
