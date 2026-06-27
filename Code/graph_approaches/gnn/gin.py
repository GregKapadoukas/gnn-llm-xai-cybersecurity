import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv  # type: ignore


def make_gin_mlp(in_features: int, out_features: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.BatchNorm1d(out_features),
        nn.Dropout(dropout),
        nn.Linear(out_features, out_features),
    )


class GIN(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        number_features: int,
        hidden_features: int,
        batch_size: int,
        dropout: float,
        num_classes: int,
        aggregator_type: str,
        learn_eps: bool,
        device,
    ):
        super(GIN, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.number_features = number_features
        self.hidden_features = hidden_features
        self.batch_size = batch_size
        self.num_classes = num_classes
        feature_scale = torch.ones(number_features, dtype=torch.float32)
        if number_features >= 4:
            feature_scale[:4] = torch.tensor(
                [65535.0, 65535.0, 65535.0, 2047.0],
                dtype=torch.float32,
            )
        self.register_buffer(
            "feature_scale", feature_scale.view(1, -1), persistent=False
        )

        self.conv1 = GINConv(
            make_gin_mlp(number_features, hidden_features, dropout),
            aggregator_type=aggregator_type,
            learn_eps=learn_eps,
        )
        self.conv2 = GINConv(
            make_gin_mlp(hidden_features, hidden_features, dropout),
            aggregator_type=aggregator_type,
            learn_eps=learn_eps,
        )

        self.dropout = nn.Dropout(dropout, inplace=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_features)
        self.linear = nn.Linear(hidden_features * number_nodes, num_classes)

    def forward(self, graphs):
        batched_graph = dgl.batch(graphs).to(self.device)
        node_features = batched_graph.ndata["feature"].float()
        node_features = torch.clamp(node_features / self.feature_scale, 0.0, 1.0)

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
