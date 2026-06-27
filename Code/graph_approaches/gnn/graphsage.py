import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv  # type: ignore


class GraphSAGE(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        number_features: int,
        hidden_features: int,
        batch_size: int,
        dropout: float,
        num_classes: int,
        aggregator_type: str,
        device,
    ):
        super(GraphSAGE, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.number_features = number_features
        self.hidden_features = hidden_features
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = SAGEConv(
            number_features,
            hidden_features,
            aggregator_type,
        )
        self.conv2 = SAGEConv(
            hidden_features,
            hidden_features,
            aggregator_type,
        )

        self.dropout = nn.Dropout(dropout, inplace=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_features)
        self.linear = nn.Linear(hidden_features * number_nodes, num_classes)

    def forward(self, graphs):
        batched_graph = dgl.batch(graphs).to(self.device)
        node_features = batched_graph.ndata["feature"].float()

        conv1_output = F.relu(self.conv1(batched_graph, node_features))
        conv1_output = self.dropout(conv1_output)

        out = self.conv2(batched_graph, conv1_output)
        out = out.reshape(len(graphs), self.number_nodes, self.hidden_features)

        out = out.transpose(1, 2)
        out = self.batchnorm1(out)
        out = out.transpose(1, 2)

        out = F.relu(out)
        out = self.dropout(out)
        out = out.reshape(len(graphs), -1)
        out = self.linear(out)
        return out
