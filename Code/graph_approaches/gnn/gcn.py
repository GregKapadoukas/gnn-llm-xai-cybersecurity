import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        number_features: int,
        batch_size: int,
        dropout: float,
        num_classes: int,
        device,
    ):
        super(GCN, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.number_features = number_features
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = GraphConv(
            number_features,
            number_features,
            norm="both",
            weight=True,
            bias=True,
            allow_zero_in_degree=False,
        )
        self.dropout = nn.Dropout(dropout, inplace=False)

        self.conv2 = GraphConv(
            number_features,
            number_features,
            norm="both",
            weight=True,
            bias=True,
            allow_zero_in_degree=False,
        )

        self.batchnorm1 = nn.BatchNorm1d(num_features=number_features)

        self.linear = nn.Linear(number_features * number_nodes, num_classes)

    def forward(self, graphs):
        out = []
        for graph in graphs:
            # Compute first graph convolution layer
            conv1_output = F.relu(self.conv1(graph, graph.ndata["feature"]))
            conv1_output = self.dropout(conv1_output)

            # Compute second graph convolution layer
            out.append(self.conv2(graph, conv1_output))

        # Stack outputs into batch output
        out = torch.stack(out, dim=0)

        # Transpose dim 1 and 2 for BatchNorm1d since it needs different format
        out = out.transpose(1, 2)

        # Batch normalize attention and residual connections
        out = self.batchnorm1(out)

        # Transpose back
        out = out.transpose(1, 2)

        # Reshape and pass through linear layer to get class per graph
        out = out.reshape(self.batch_size, -1)
        out = self.linear(out)
        return out
