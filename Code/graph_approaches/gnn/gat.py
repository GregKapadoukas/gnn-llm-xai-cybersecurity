import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv  # type: ignore


class GAT(nn.Module):
    def __init__(
        self,
        number_nodes: int,
        number_features: int,
        out_feats: int,
        batch_size: int,
        dropout: float,
        num_classes: int,
        num_heads: int,
        device,
    ):
        super(GAT, self).__init__()

        self.device = device
        self.number_nodes = number_nodes
        self.number_features = number_features
        self.out_feats = out_feats
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        feature_scale = torch.ones(number_features, dtype=torch.float32)
        if number_features >= 4:
            feature_scale[:4] = torch.tensor(
                [65535.0, 65535.0, 65535.0, 2047.0],
                dtype=torch.float32,
            )
        self.register_buffer(
            "feature_scale", feature_scale.view(1, -1), persistent=False
        )

        self.conv1 = GATConv(
            number_features,
            out_feats,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=F.elu,
            allow_zero_in_degree=True,
        )
        self.conv2 = GATConv(
            out_feats * num_heads,
            out_feats,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
            allow_zero_in_degree=True,
        )

        self.dropout = nn.Dropout(dropout, inplace=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_feats)
        self.linear = nn.Linear(out_feats * number_nodes, num_classes)

    def forward(self, graphs):
        graphs = [dgl.add_self_loop(dgl.remove_self_loop(graph)) for graph in graphs]
        batched_graph = dgl.batch(graphs).to(self.device)
        node_features = batched_graph.ndata["feature"].float()
        node_features = torch.clamp(node_features / self.feature_scale, 0.0, 1.0)

        conv1_output = self.conv1(batched_graph, node_features).flatten(1)
        conv1_output = self.dropout(conv1_output)

        out = self.conv2(batched_graph, conv1_output).mean(1)
        out = out.reshape(len(graphs), self.number_nodes, self.out_feats)

        out = out.transpose(1, 2)
        out = self.batchnorm1(out)
        out = out.transpose(1, 2)

        out = F.elu(out)
        out = self.dropout(out)
        out = out.reshape(len(graphs), -1)
        out = self.linear(out)
        return out
