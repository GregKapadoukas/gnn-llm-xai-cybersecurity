"""
Graph Transformer Structure based on: "A Generalization of Transformer Networks to Graphs: Vijay Parkash Dwivedi, Xavier Bresson":

1. Attention mechanism is a function of the neighborhood connectivity for each node in the Graph
2. The positional encoding is represented by the Laplacian eigenvectors, which naturally generalize the sinusoidal positional encodings often used in NPL.
    Laplacian PE: factorization of graph Laplacian matrix: Δ = I - D^(-1/2)*A*D^(-1/2) = (U^T)*Λ*U
    A: n*n adjacency matrix,
    D: degree matrix,
    Λ, U: eigenvalues and eigenvectors respectively
    k smallest non-trivial eigenvectors of a node is it's positional encoding denoted by λi for node i
3. The layer normalization is replaced by a batch normalization layer, which provides faster training and better generalization performance

Input: For a graph G with node features αi ε R^(dn x 1) for each node i the input features are passed via a linear projection to embed these to d-dimensional hidden features h_hat_i_0 = A_0 + a_0
    where A_0 ε R(d x dn) is the parameter of the linear projection layer.
    We now embed the pre-computed node positional encodings of dim k via a linear projection: λ_i_0 = C_0 * λ_i + c_0; h_i_0 = h_hat_i+0 + λ_i_0
    where C_0 ε R(d x k) and c_0 ε R_d.

"""

import dgl
import torch
import torch.nn.functional as F
from torch import nn, where


# LaplacianGraphEmbeddings use linearly projected Laplacian positional encoding along with linearly projected node features
class LaplacianGraphEmbeddings(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        node_features_size: int,
        number_eigenvectors: int,
        embedding_size: int,
        device,
    ):
        super(LaplacianGraphEmbeddings, self).__init__()
        self.number_nodes = number_nodes
        self.number_eigenvectors = number_eigenvectors
        self.embedding_size = embedding_size
        self.positional_encoding_linear_layer = nn.Linear(
            number_eigenvectors, embedding_size
        )
        self.node_features_linear_layer = nn.Linear(node_features_size, embedding_size)
        self.device = device

    def forward(self, graphs):
        node_embeddings = []
        adjacency_matrices = []
        for graph in graphs:
            # Compute Laplacian Positional Encoding (laplacian eigenvectors) and their linear projection
            positional_encoding = dgl.lap_pe(graph, self.number_eigenvectors)
            positional_encoding = positional_encoding.to(self.device)  # type: ignore
            positional_encoding_projected = self.positional_encoding_linear_layer(
                positional_encoding
            )
            # Linearly project graph features
            node_features_projected = self.node_features_linear_layer(
                graph.ndata["feature"].float()
            )

            # Create final graph embeddings
            single_graph_embeddings = (
                node_features_projected + positional_encoding_projected
            )
            node_embeddings.append(single_graph_embeddings)
            adjacency_matrices.append(graph.adjacency_matrix().to_dense())
        node_embeddings = torch.stack((node_embeddings), dim=0)
        adjacency_matrices = torch.stack((adjacency_matrices), dim=0)
        return node_embeddings, adjacency_matrices


class MultiheadGraphAttention(nn.Module):
    def __init__(self, number_nodes: int, embedding_size: int, num_heads: int, device):
        super(MultiheadGraphAttention, self).__init__()
        self.number_nodes = number_nodes
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.device = device
        assert (
            embedding_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"
        self.d_k = embedding_size // num_heads

        # Use a single parameter for queries, keys, and values but in a larger tensor
        self.query_matrices = nn.Parameter(
            torch.Tensor(num_heads, embedding_size, self.d_k)
        )
        self.key_matrices = nn.Parameter(
            torch.Tensor(num_heads, embedding_size, self.d_k)
        )
        self.value_matrices = nn.Parameter(
            torch.Tensor(num_heads, embedding_size, self.d_k)
        )
        self.output_layer = nn.Linear(embedding_size, embedding_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_matrices)
        nn.init.xavier_uniform_(self.key_matrices)
        nn.init.xavier_uniform_(self.value_matrices)

    def forward(self, node_embeddings, adjacency_matrices):
        batch_size, num_nodes, _ = node_embeddings.shape

        # Expand node_embeddings for multi-head processing
        node_embeddings_expanded = node_embeddings.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )

        # Compute every Q, K, v
        Q = torch.einsum(
            "bhnd,hde->bhne", node_embeddings_expanded, self.query_matrices
        )
        K = torch.einsum("bhnd,hde->bhne", node_embeddings_expanded, self.key_matrices)
        V = torch.einsum(
            "bhnd,hde->bhne", node_embeddings_expanded, self.value_matrices
        )

        # Calculate every attention score
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # Apply adjacency mask so that score only matters to neighboring nodes
        adjacency_matrices_expanded = adjacency_matrices.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )
        attention_scores = torch.where(
            adjacency_matrices_expanded > 0,
            attention_scores,
            torch.tensor(float("-inf")).to(attention_scores.device),
        )

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Dot product the attention weights to the values
        head_outputs = torch.matmul(attention_weights, V)

        # Apply final linear transformation
        concatenated_outputs = head_outputs.reshape(batch_size, num_nodes, -1)
        multihead_attentions = self.output_layer(concatenated_outputs)

        return multihead_attentions


class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        embedding_size: int,
        feedforward_scaling: int,
        num_heads: int,
        dropout: float,
        device,
    ):
        super(GraphEncoderLayer, self).__init__()
        # Create linear layers to compute Query, Key and Value
        self.query_layer = nn.Linear(embedding_size, embedding_size)
        self.key_layer = nn.Linear(embedding_size, embedding_size)
        self.value_layer = nn.Linear(embedding_size, embedding_size)
        self.device = device

        # Multi-head self-attention layer
        self.multihead_attention = MultiheadGraphAttention(
            number_nodes, embedding_size, num_heads, self.device
        )

        self.dropout1 = nn.Dropout(dropout, inplace=True)

        # First batch normalization layer
        self.batchnorm1 = nn.BatchNorm1d(num_features=embedding_size)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        # Feed forward layer (with added dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_scaling * embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_scaling * embedding_size, embedding_size),
        )

        # Second batch normalization layer
        self.batchnorm2 = nn.BatchNorm1d(num_features=embedding_size)

    def forward(
        self, embeddings_or_encoded_output, adjacency_matrices, residual_connections_1
    ):
        # Compute attention
        attention = self.multihead_attention(
            embeddings_or_encoded_output, adjacency_matrices
        )

        self.dropout1(attention)

        # Transpose dim 1 and 2 for BatchNorm1d since it needs different format
        attention = attention.transpose(1, 2)

        # Batch normalize attention and residual connections
        attention_norm1 = self.batchnorm1(attention + residual_connections_1)
        self.dropout2(attention_norm1)

        # Transpose back to original shape
        residual_connections_2 = attention_norm1
        attention_norm1 = attention_norm1.transpose(1, 2)

        # Feed-forward the output and residual connections
        feedforward = self.feedforward(attention_norm1)

        # Transpose dim 1 and 2 for BatchNorm1d since it needs different format
        # (residual connections 2 already transposed)
        feedforward = feedforward.transpose(1, 2)

        # Batch normalize the output
        out = self.batchnorm2(feedforward + residual_connections_2)

        # Transpose back to original shape
        out = out.transpose(1, 2)
        return out


class GraphTransformer(nn.Module):
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
        num_classes: int,
        device,
    ):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.num_classes = num_classes

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

        self.dropout = nn.Dropout(dropout, inplace=True)

        # Create final linear layer for the classification
        self.linear = nn.Linear(number_nodes * embedding_size, num_classes)
        # Softmax the output for the classification is not needed because of logits of CrossEntropyLoss

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
        self.dropout(encoded_outputs)

        # Pass them to the linear layer
        encoded_outputs = encoded_outputs.reshape(len(graphs), -1)
        out = self.linear(encoded_outputs)
        return out
