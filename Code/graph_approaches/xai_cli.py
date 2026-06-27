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
    IGWrapper,
    SHAPWrapper,
    collate_graphs,
    compute_ig_metrics,
    compute_shap_metrics,
    explain_with_integrated_gradients,
    explain_with_shap,
    summarize_integrated_gradients,
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
parser.add_argument(
    "--ig_plot",
    default="xai_outputs/ig_quadrant_attributions.png",
    help="path for the Integrated Gradients quadrant graph; use '' to skip saving",
)
parser.add_argument(
    "--ig_target_class",
    default="true",
    help="IG class to explain: 'true', 'predicted', 'all', or an integer class index",
)
parser.add_argument(
    "--shap_plot",
    default="xai_outputs/shap_quadrant_attributions.png",
    help="path for the local SHAP quadrant graph; use '' to skip saving",
)
parser.add_argument(
    "--shap_plot_class",
    default="all",
    help="class index to visualize for local SHAP, or 'all'; defaults to all classes",
)
parser.add_argument(
    "--summary_top_k",
    type=int,
    default=10,
    help="number of top attribution contributors to print/save; use 0 to disable summaries",
)
parser.add_argument(
    "--summary_level",
    choices=("node_feature", "node", "feature"),
    default="node_feature",
    help="summary granularity: node_feature, node, or feature",
)
parser.add_argument(
    "--ig_summary_graphs",
    type=int,
    default=10,
    help="number of graphs used for the aggregate IG top-attribution summary",
)
parser.add_argument(
    "--ig_summary_plot",
    default="xai_outputs/ig_top_attributions.png",
    help="path for the aggregate IG top-attribution bar chart; use '' to skip saving",
)
parser.add_argument(
    "--ig_summary_csv",
    default="xai_outputs/ig_top_attributions.csv",
    help="path for the aggregate IG top-attribution CSV; use '' to skip saving",
)
parser.add_argument(
    "--shap_global_summary_plot",
    default="xai_outputs/shap_global_top_attributions.png",
    help="path for SHAP global top-attribution bar charts; use '' to skip saving",
)
parser.add_argument(
    "--shap_global_summary_csv",
    default="xai_outputs/shap_global_top_attributions.csv",
    help="path for the SHAP global top-attribution CSV; use '' to skip saving",
)
parser.add_argument(
    "--shap_local_summary_plot",
    default="xai_outputs/shap_local_top_attributions.png",
    help="path for SHAP local top-attribution bar charts; use '' to skip saving",
)
parser.add_argument(
    "--shap_local_summary_csv",
    default="xai_outputs/shap_local_top_attributions.csv",
    help="path for the SHAP local top-attribution CSV; use '' to skip saving",
)
args = parser.parse_args()
device = torch.device(args.device)

if args.shap_plot_class is not None and args.shap_plot_class != "all":
    args.shap_plot_class = int(args.shap_plot_class)


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
    # loadGraphDataset.oneHotEncode expects 1-based IDs and then subtracts 1.
    # Model/display class indices remain 0-based: Benign=0, Attack=1.
    mapping = {lbl: (1 if lbl == "Benign" else 2) for lbl in all_classes}
    final_num_classes = 2
    class_names = {0: "Benign", 1: "Attack"}
else:
    ordered = sorted(all_classes, key=lambda x: (x != "Benign", x))
    # loadGraphDataset.oneHotEncode expects 1-based IDs and then subtracts 1.
    mapping = {lbl: idx + 1 for idx, lbl in enumerate(ordered)}
    final_num_classes = len(mapping)
    class_names = {idx - 1: label for label, idx in mapping.items()}


def class_label(class_idx):
    return class_names.get(int(class_idx), f"Class {class_idx}")

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
    benign_idx = mapping["Benign"] - 1
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

ig_wrapper = IGWrapper(model, device)
ig_x, ig_A = collate_graphs([graph], device)
with torch.no_grad():
    logits = ig_wrapper(ig_x, ig_A)
    pred_class = logits.argmax(-1).item()
    pred_score = logits[0, pred_class].item()

if args.ig_target_class == "true":
    ig_target_classes = [true_class]
elif args.ig_target_class == "predicted":
    ig_target_classes = [pred_class]
elif args.ig_target_class == "all":
    ig_target_classes = list(range(num_classes_ckpt))
else:
    ig_target_classes = [int(args.ig_target_class)]

bad_ig_target_classes = [
    class_idx for class_idx in ig_target_classes if not 0 <= class_idx < num_classes_ckpt
]
if bad_ig_target_classes:
    raise ValueError(
        "--ig_target_class must be 'true', 'predicted', 'all', "
        f"or in [0, {num_classes_ckpt - 1}]"
    )


def class_output_path(path, class_idx):
    if path is None or len(ig_target_classes) == 1:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_class{class_idx}{ext}"

print(f"Selected graph index: {idx}")
print(f"True class: {true_class} ({class_label(true_class)})")
print(f"Predicted class: {pred_class} ({class_label(pred_class)}, logit={pred_score:.4f})")
print(f"Prediction correct: {pred_class == true_class}")
print(
    "IG target classes being explained: "
    + ", ".join(f"{c} ({class_label(c)})" for c in ig_target_classes)
)

for ig_target_class in ig_target_classes:
    ig_plot_title = (
        "Integrated Gradients | "
        f"True: {class_label(true_class)} | "
        f"Predicted: {class_label(pred_class)} | "
        f"Attributed: {class_label(ig_target_class)}"
    )

    explain_with_integrated_gradients(
        model,
        graph,
        target_class=ig_target_class,
        steps=100,
        attribute_names=node_attribute_names,
        plot_path=class_output_path(args.ig_plot or None, ig_target_class),
        plot_title=ig_plot_title,
        class_name=class_label(ig_target_class),
        summary_top_k=0,
    )

ig_summary_graphs = graphs[:max(0, args.ig_summary_graphs)]
if args.summary_top_k > 0 and ig_summary_graphs:
    summarize_integrated_gradients(
        model,
        ig_summary_graphs,
        target_classes=ig_target_classes,
        steps=100,
        attribute_names=node_attribute_names,
        class_name_fn=class_label,
        summary_top_k=args.summary_top_k,
        summary_level=args.summary_level,
        summary_plot_path=args.ig_summary_plot or None,
        summary_csv_path=args.ig_summary_csv or None,
    )

# ─────────── 6.  SHAP global explanation ────────────
print("\nSHAP")
background = graphs[10:30]
sample = graphs[:10]

# GLOBAL explanation of 10 graphs
global_df = explain_with_shap(
    model,
    background,
    sample_graphs=graphs[:10],
    attribute_names=node_attribute_names,  # ["Src Port", ...]
    mode="global",
    plot_title_context=class_label,
    summary_top_k=args.summary_top_k,
    summary_level=args.summary_level,
    summary_plot_path=args.shap_global_summary_plot or None,
    summary_csv_path=args.shap_global_summary_csv or None,
)  # optional: auto would do the same

# LOCAL explanation of a single graph
shap_local_graphs = [graphs[0]]
shap_true_classes = [labels[0].argmax().item()]
shap_pred_classes = []
for shap_graph in shap_local_graphs:
    shap_x, shap_A = collate_graphs([shap_graph], device)
    with torch.no_grad():
        shap_logits = ig_wrapper(shap_x, shap_A)
        shap_pred_classes.append(shap_logits.argmax(-1).item())


def shap_plot_title(local_idx, attributed_class):
    true_label = class_label(shap_true_classes[local_idx])
    pred_label = class_label(shap_pred_classes[local_idx])
    attr_label = class_label(attributed_class)
    return (
        f"SHAP | Graph {local_idx} | "
        f"True: {true_label} | "
        f"Predicted: {pred_label} | "
        f"Attributed: {attr_label}"
    )


local_df = explain_with_shap(
    model,
    background,
    sample_graphs=shap_local_graphs,
    attribute_names=node_attribute_names,
    mode="local",
    plot_path=args.shap_plot or None,
    plot_class=args.shap_plot_class,
    plot_title_context=class_label,
    plot_title_builder=shap_plot_title,
    summary_top_k=args.summary_top_k,
    summary_level=args.summary_level,
    summary_plot_path=args.shap_local_summary_plot or None,
    summary_csv_path=args.shap_local_summary_csv or None,
)  # or just omit mode if only 1 graph

# ─────────── 7.  Quantitative metrics  ────────────
metrics_num_graphs = 5
eval_graphs = graphs[:metrics_num_graphs]  # any subset you like

print(f"\n=== XAI metrics (IG) on {metrics_num_graphs} graphs ===")
print(compute_ig_metrics(model, eval_graphs, noise_std=0.01))

print(f"\n=== XAI metrics (SHAP) on the same {metrics_num_graphs} graphs ===")
background = graphs[10:30]  # 20-graph reference
print(compute_shap_metrics(model, background, eval_graphs, noise_std=0.01))
