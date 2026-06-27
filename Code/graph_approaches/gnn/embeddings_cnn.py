import math

import torch
import torch.nn as nn

from graph_transformer.graph_transformer import LaplacianGraphEmbeddings


class EmbeddingsCNN(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        batch_size: int,
        node_features_size: int,
        number_eigenvectors: int,
        embedding_size: int,
        dropout: float,
        num_classes: int,
        device,
    ):
        super(EmbeddingsCNN, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Laplacian graph embeddings layer
        self.laplacian_embed_layer = LaplacianGraphEmbeddings(
            number_nodes,
            batch_size,
            node_features_size,
            number_eigenvectors,
            embedding_size,
            device,
        )
        self.conv1 = nn.Conv1d(
            embedding_size, math.floor(embedding_size / 2), kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.conv2 = nn.Conv1d(
            math.floor(embedding_size / 2), num_classes, kernel_size=3, padding=1
        )

    def forward(self, graphs):
        node_embeddings, _ = self.laplacian_embed_layer(graphs)
        node_embeddings = torch.stack(node_embeddings, dim=0)

        # Compute first convolutional layer
        node_embeddings_permuted = node_embeddings.permute(0, 2, 1)
        conv1_output = self.conv1(node_embeddings_permuted)

        self.dropout(conv1_output)

        # Compute second convolutional layer
        conv2_output = self.conv2(conv1_output)

        # Return output
        conv2_output_permuted = conv2_output.permute(0, 2, 1)
        out = conv2_output_permuted.mean(dim=1)
        return out
