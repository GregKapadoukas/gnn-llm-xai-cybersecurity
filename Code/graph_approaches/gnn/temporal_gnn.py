import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN


class TemporalGNN(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        number_features: int,
        batch_size: int,
        dropout: float,
        num_classes: int,
        num_time_steps: int,
        device,
    ):
        super(TemporalGNN, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.number_features = number_features
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps

        self.temporalgnn = DCRNN(
            in_channels=number_features, out_channels=number_features, K=5, bias=True
        )

        self.dropout = nn.Dropout(dropout, inplace=False)

        self.layernorm = nn.LayerNorm(normalized_shape=number_features)

        self.linear = nn.Linear(
            num_time_steps * number_features * number_nodes, num_classes
        )

    def forward(self, graphs):
        out = []
        for graph in graphs:
            conv_output = []
            for snapshot in graph:
                # Compute single graph convolution layer
                X = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                conv_output.append(self.temporalgnn(X, edge_index))
            conv_output = torch.stack(conv_output, dim=0).to(self.device)
            conv_output = F.relu(conv_output)
            conv_output = self.dropout(conv_output)
            out.append(conv_output)

        # Stack outputs into batch output
        out = torch.stack(out, dim=0).to(self.device)

        # Batch normalize attention and residual connections
        out = self.layernorm(out)

        # Reshape and pass through linear layer to get class per graph
        out = out.reshape(self.batch_size, -1)
        out = self.linear(out)
        return out
