#!/usr/bin/env python
# xai_cli.py – auto-param recovery + DGL loader + IG + SHAP
import argparse
import os
import random
import re

import pandas as pd
import shap
import torch

from gnn.graph_transformer import GraphTransformer
from graph_dataset.graph_dataset import loadGraphDataset  # ← your helper
from xai.xai_utils import (
    SHAPWrapper,
    collate_graphs,
    compute_ig_metrics,
    compute_shap_metrics,
    explain_with_integrated_gradients,
    explain_with_shap,
)

# ─────────────────────────────  CLI  ──────────────────────────────
parser = argparse.ArgumentParser("Explain GraphTransformer predictions")
parser.add_argument("--ckpt", required=True, help="checkpoint .pt file")
parser.add_argument("--graphs_dir", required=True, help="folder with *.bin graphs")
parser.add_argument("--files_per_class", type=int, default=50)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument(
    "--binary", action="store_true", help="collapse to Benign (0) vs Attack (1)"
)
args = parser.parse_args()
device = torch.device(args.device)


# ─────────── 1.  Recover hyper-parameters from the checkpoint ────────────
state = torch.load(args.ckpt, map_location="cpu")


def shp(k):
    return state[k].shape


embedding_size, number_eigenvectors = shp(
    "laplacian_embed_layer.positional_encoding_linear_layer.weight"
)
node_features_size = shp("laplacian_embed_layer.node_features_linear_layer.weight")[1]
num_heads = shp("graph_encoder.0.multihead_attention.query_matrices")[0]
feedforward_scaling = shp("graph_encoder.0.feedforward.0.weight")[0] // embedding_size
num_layers = (
    max(int(m.group(1)) for k in state if (m := re.match(r"graph_encoder\.(\d+)\.", k)))
    + 1
)
num_classes_ckpt = shp("linear.weight")[0]
number_nodes = shp("linear.weight")[1] // embedding_size
dropout = 0.5


# ─────────── 2.  Build one_hot_mapping automatically ────────────
def extract_class(fn: str) -> str:
    """file name → class label (your naming scheme: *_<class>.bin)"""
    return fn.split(".")[0].split("_")[1]


all_files = os.listdir(args.graphs_dir)
all_classes = {extract_class(f) for f in all_files}

if args.binary:
    # Benign = 0 ; every other class maps to Attack = 1 (later we convert)
    mapping = {lbl: (0 if lbl == "Benign" else 1) for lbl in all_classes}
    final_num_classes = 2
else:
    ordered = sorted(all_classes, key=lambda x: (x != "Benign", x))
    mapping = {lbl: idx for idx, lbl in enumerate(ordered)}
    final_num_classes = len(mapping)

if final_num_classes != num_classes_ckpt:
    print(
        f"⚠  Checkpoint has {num_classes_ckpt} classes but "
        f"{'binary' if args.binary else 'dataset'} setup yields {final_num_classes}. "
        "Loading will still work; just be sure this is intended."
    )

# ─────────── 3.  Instantiate model & load weights ────────────
model = (
    GraphTransformer(
        number_nodes,
        node_features_size,
        number_eigenvectors,
        embedding_size,
        feedforward_scaling,
        num_heads,
        num_layers,
        dropout,
        num_classes_ckpt,
        device,
    )
    .to(device)
    .eval()
)
model.load_state_dict(state, strict=True)

print("\n✔  Classes:")
print(mapping)

print("\n✔  Hyper-parameters:")
for k, v in dict(
    number_nodes=number_nodes,
    node_features_size=node_features_size,
    number_eigenvectors=number_eigenvectors,
    embedding_size=embedding_size,
    feedforward_scaling=feedforward_scaling,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes_ckpt,
    dropout=dropout,
).items():
    print(f"   • {k:24s}: {v}")

# ─────────── 4.  Load graphs via DGL helper ────────────
node_attribute_names = ["Src Port", "Dst Port", "Length", "Proto/Flag"]  # F = 4
feature_names = [
    f"Node {i} Feature {node_attribute_names[j]}"
    for i in range(number_nodes)
    for j in range(len(node_attribute_names))
]
graphs, labels = loadGraphDataset(args.graphs_dir, mapping, args.files_per_class)

# If we requested binary but model is multiclass (or vice-versa), we still
# use the correct *target label* when choosing a graph for IG.
if args.binary:
    benign_idx = mapping["Benign"]
    labels_bin = torch.zeros(len(labels), 2, dtype=labels.dtype)
    labels_bin[:, 0] = labels[:, benign_idx]  # Benign
    labels_bin[:, 1] = 1 - labels[:, benign_idx]  # Attack
    labels = labels_bin
print(f"\n✔  Loaded {len(graphs)} graphs")

# ─────────── 5.  Integrated Gradients on a random graph ────────────
print("\nIntegrated Gradients")
idx = random.randrange(len(graphs))
graph = graphs[idx]
true_class = labels[idx].argmax().item()

explain_with_integrated_gradients(
    model,
    graph,
    target_class=true_class,
    steps=100,
    attribute_names=node_attribute_names,
)

# ─────────── 6.  SHAP global explanation ────────────
print("\nSHAP")

num_background_graphs = 200
num_sample_graphs = 3000

background = graphs[:num_background_graphs]
sample = graphs[num_background_graphs:num_background_graphs + num_sample_graphs]

print(f"The number of background graphs is {num_background_graphs}")
print(f"The number of sample graphs is {num_sample_graphs}")

# GLOBAL explanation of num_sample_graphs graphs
global_df = explain_with_shap(
    model,
    background,
    sample_graphs=sample,
    attribute_names=node_attribute_names,  # ["Src Port", ...]
    mode="global",
)  # optional: auto would do the same

# LOCAL explanation of a single graph
local_df = explain_with_shap(
    model,
    background,
    sample_graphs=sample[0],
    attribute_names=node_attribute_names,
    mode="local",
)  # or just omit mode if only 1 graph

# ─────────── 7.  Quantitative metrics  ────────────
print(f"\n=== XAI metrics (IG) on {len(sample)} graphs ===")
print(compute_ig_metrics(model, sample, noise_std=0.01))

print(f"\n=== XAI metrics (SHAP) on the same {len(sample)} graphs ===")
print(compute_shap_metrics(model, background, sample, noise_std=0.01))
