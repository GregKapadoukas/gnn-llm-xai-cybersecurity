# xai_utils.py
"""
XAI helpers for the GraphTransformer model defined in graph_transformer.py
=========================================================================

Functions
---------
collate_graphs(...)                   → tensors ready for IG / SHAP
summarize_attribution_df(...)         → ranked top node/feature contributors
IGWrapper(nn.Module)    → makes the model look like f(x, A) → logits
explain_with_integrated_gradients(...)→ node-level IG on ONE graph
explain_with_shap(...)                → global feature importances

Run `python xai_utils.py --help` for a CLI demo.
"""
from __future__ import annotations

from pathlib import Path
import ipaddress
import re

import dgl
import numpy as np
import pandas as pd
import shap
import torch
from captum.attr import IntegratedGradients
from captum.metrics import (infidelity, infidelity_perturb_func_decorator,
                            sensitivity_max)
from tqdm import tqdm


_POSITIVE_COLOR = np.array([31, 142, 64]) / 255.0
_NEGATIVE_COLOR = np.array([203, 24, 29]) / 255.0
_NEUTRAL_COLOR = np.array([242, 242, 242]) / 255.0
_TCP_FLAG_BITS = (
    ("F", 9),
    ("S", 8),
    ("R", 7),
    ("P", 6),
    ("A", 5),
    ("U", 4),
    ("E", 3),
    ("C", 2),
    ("N", 1),
)


def _attribution_color(value, max_abs):
    """Map signed attribution to a red/green color with intensity by magnitude."""
    if max_abs <= 0:
        return _NEUTRAL_COLOR

    strength = min(abs(float(value)) / max_abs, 1.0)
    target = _POSITIVE_COLOR if value >= 0 else _NEGATIVE_COLOR
    return _NEUTRAL_COLOR * (1.0 - strength) + target * strength


def _graph_edges(graph):
    """Return non-self-loop edges as plain Python integer tuples."""
    src, dst = graph.edges()
    edges = []
    for u, v in zip(src.cpu().tolist(), dst.cpu().tolist()):
        if u != v:
            edges.append((int(u), int(v)))
    return edges


def _graph_layout(num_nodes, edges, seed=7):
    """Use NetworkX spring layout when available, falling back to a circle."""
    try:
        import networkx as nx

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        nx_graph.add_edges_from(edges)
        if edges:
            return nx.spring_layout(nx_graph, seed=seed, k=1.3 / np.sqrt(num_nodes))
    except ImportError:
        pass

    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}


def _spread_layout(pos, min_distance, iterations=250):
    """Push nodes apart so rendered circles do not overlap."""
    pos = {node: np.array(point, dtype=float) for node, point in pos.items()}
    nodes = list(pos)
    if len(nodes) < 2:
        return pos

    for _ in range(iterations):
        moved = False
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                delta = pos[node_b] - pos[node_a]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue

                if distance == 0:
                    angle = (i + 1) * np.pi / max(len(nodes), 1)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    distance = 1e-9
                else:
                    direction = delta / distance

                shift = direction * ((min_distance - distance) / 2.0)
                pos[node_a] -= shift
                pos[node_b] += shift
                moved = True

        if not moved:
            break

    return pos


def _decode_protocol_flags(value):
    """Best-effort inverse of the protocol/flag decimal encoding."""
    if not np.isfinite(value):
        return str(value)

    int_value = int(round(float(value)))
    if abs(float(value) - int_value) > 1e-6:
        return f"{value:.3g}"
    if int_value == 0:
        return "UDP"

    flags = [flag for flag, bit in _TCP_FLAG_BITS if int_value & (1 << bit)]
    return "TCP" + ("+" + "".join(flags) if flags else "")


def _decode_ipv4(value):
    """Best-effort inverse for integer-encoded IPv4 values."""
    if not np.isfinite(value):
        return str(value)

    int_value = int(round(float(value)))
    if abs(float(value) - int_value) > 1e-6:
        return f"{value:.3g}"
    if not 0 <= int_value <= 0xFFFFFFFF:
        return str(int_value)

    ip = str(ipaddress.ip_address(int_value))
    parts = ip.split(".")
    return ".".join(parts[:2]) + "\n" + ".".join(parts[2:])


def _format_node_attribute(value, feature_name):
    """Compact labels that can fit inside small node quadrants."""
    if "ip" in feature_name.lower():
        return _decode_ipv4(value)
    if "proto" in feature_name.lower() or "flag" in feature_name.lower():
        return _decode_protocol_flags(value)

    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3g}"


def plot_quadrant_attributions(
    graph,
    attribution_df,
    output_path,
    *,
    title=None,
    figsize=(14, 10),
    dpi=180,
    show_node_attributes=True,
):
    """
    Save a graph visualization where each node is split into four feature quadrants.

    Positive attributions are green, negative attributions are red, and stronger
    absolute attribution values produce stronger color intensity. The function is
    intended for four node features, but it will plot up to the first four columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Wedge

    mat = attribution_df.to_numpy(dtype=float)
    num_nodes, num_features = mat.shape
    if num_features < 1:
        raise ValueError("attribution_df must contain at least one feature column")

    feature_names = list(attribution_df.columns[:4])
    mat = mat[:, :4]
    feature_values = graph.ndata["feature"].detach().cpu().numpy()[:, :4]
    max_abs = float(np.max(np.abs(mat))) if mat.size else 0.0
    edges = _graph_edges(graph)
    pos = _graph_layout(num_nodes, edges)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    radius = 0.13 if num_nodes > 16 else 0.17
    pos = _spread_layout(pos, min_distance=radius * 2.55)

    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#9e9e9e", linewidth=1.0, alpha=0.5, zorder=1)

    quadrant_angles = [(0, 90), (90, 180), (180, 270), (270, 360)]

    for node_idx in range(num_nodes):
        x, y = pos[node_idx]
        for feat_idx, (theta1, theta2) in enumerate(quadrant_angles):
            value = mat[node_idx, feat_idx] if feat_idx < mat.shape[1] else 0.0
            wedge = Wedge(
                (x, y),
                radius,
                theta1,
                theta2,
                facecolor=_attribution_color(value, max_abs),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(wedge)

            if show_node_attributes and feat_idx < feature_values.shape[1]:
                theta = np.deg2rad((theta1 + theta2) / 2)
                label_x = x + np.cos(theta) * radius * 0.48
                label_y = y + np.sin(theta) * radius * 0.48
                label = _format_node_attribute(
                    feature_values[node_idx, feat_idx],
                    feature_names[feat_idx],
                )
                ax.text(
                    label_x,
                    label_y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.4 if num_nodes > 16 else 8.0,
                    color="#111111",
                    zorder=5,
                )

        outline = Circle(
            (x, y),
            radius,
            facecolor="none",
            edgecolor="#303030",
            linewidth=0.9,
            zorder=4,
        )
        ax.add_patch(outline)
        ax.text(
            x,
            y - radius * 1.45,
            str(node_idx),
            ha="center",
            va="top",
            fontsize=9 if num_nodes > 16 else 10,
        )

    if title:
        ax.set_title(title, fontsize=16, pad=18)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_POSITIVE_COLOR,
               markeredgecolor="none", markersize=9, label="Positive attribution"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_NEGATIVE_COLOR,
               markeredgecolor="none", markersize=9, label="Negative attribution"),
    ]
    quadrant_labels = [
        Line2D([0], [0], color="none", label=f"Q{i + 1}: {name}")
        for i, name in enumerate(feature_names)
    ]
    ax.legend(
        handles=legend_handles + quadrant_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=10,
    )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = radius * 4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_node_index(node_label, fallback):
    """Extract a numeric node index from labels like 'Node 7'."""
    match = re.search(r"\d+", str(node_label))
    return int(match.group(0)) if match else int(fallback)


def _direction_label(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _class_display_label(class_index=None, class_name=None):
    if class_name is not None:
        return str(class_name)
    if class_index is not None:
        return f"Class {class_index}"
    return "Attribution"


def _long_node_feature_attributions(
    attribution_df,
    *,
    abs_attribution_df=None,
    class_index=None,
    class_name=None,
):
    """Convert an N x F attribution table into one row per node-feature pair."""
    records = []
    class_label = _class_display_label(class_index, class_name)

    for node_pos, node_label in enumerate(attribution_df.index):
        node_idx = _parse_node_index(node_label, node_pos)
        for feature_name in attribution_df.columns:
            attribution = float(attribution_df.loc[node_label, feature_name])
            if abs_attribution_df is None:
                abs_attribution = abs(attribution)
            else:
                abs_attribution = float(abs_attribution_df.loc[node_label, feature_name])

            records.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "class_label": class_label,
                    "node": node_idx,
                    "node_label": str(node_label),
                    "feature": str(feature_name),
                    "label": f"{node_label} / {feature_name}",
                    "attribution": attribution,
                    "abs_attribution": abs_attribution,
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_attribution_df(
    attribution_df,
    *,
    abs_attribution_df=None,
    class_index=None,
    class_name=None,
    top_k=10,
    level="node_feature",
):
    """
    Rank the largest attribution contributors in a graph explanation.

    Parameters
    ----------
    attribution_df:
        DataFrame shaped (num_nodes, num_features) with signed attribution values.
    abs_attribution_df:
        Optional DataFrame with the same shape. Use this when ranking by an
        already-aggregated absolute score, e.g. global mean |SHAP|.
    level:
        "node_feature" ranks individual node-feature cells, "node" aggregates
        across features per node, and "feature" aggregates across nodes.
    """
    long_df = _long_node_feature_attributions(
        attribution_df,
        abs_attribution_df=abs_attribution_df,
        class_index=class_index,
        class_name=class_name,
    )

    if level == "node_feature":
        summary = long_df.copy()
    elif level == "node":
        summary = (
            long_df.groupby(
                ["class_label", "node", "node_label"],
                as_index=False,
            )
            .agg(
                class_index=("class_index", "first"),
                class_name=("class_name", "first"),
                attribution=("attribution", "sum"),
                abs_attribution=("abs_attribution", "sum"),
            )
        )
        summary["feature"] = "all_features"
        summary["label"] = summary["node_label"]
    elif level == "feature":
        summary = (
            long_df.groupby(
                ["class_label", "feature"],
                as_index=False,
            )
            .agg(
                class_index=("class_index", "first"),
                class_name=("class_name", "first"),
                attribution=("attribution", "sum"),
                abs_attribution=("abs_attribution", "sum"),
            )
        )
        summary["node"] = None
        summary["node_label"] = "all_nodes"
        summary["label"] = summary["feature"]
    else:
        raise ValueError("level must be one of: 'node_feature', 'node', 'feature'")

    summary = summary.sort_values(
        ["class_label", "abs_attribution"],
        ascending=[True, False],
    ).reset_index(drop=True)
    summary["rank"] = summary.groupby("class_label").cumcount() + 1

    if top_k is not None:
        summary = summary[summary["rank"] <= int(top_k)]

    summary["direction"] = summary["attribution"].map(_direction_label)

    preferred_cols = [
        "class_index",
        "class_name",
        "class_label",
        "rank",
        "label",
        "node",
        "node_label",
        "feature",
        "attribution",
        "abs_attribution",
        "direction",
    ]
    return summary[[c for c in preferred_cols if c in summary.columns]].reset_index(drop=True)


def _print_attribution_summary(summary_df, title):
    if summary_df.empty:
        print(f"\n{title}: no attributions to summarize")
        return

    display_cols = [
        "class_label",
        "rank",
        "label",
        "attribution",
        "abs_attribution",
        "direction",
    ]
    display = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
    print(f"\n{title}")
    with pd.option_context("display.max_colwidth", 80):
        print(display.round(6).to_string(index=False))


def _suffixed_output_path(output_path, suffix):
    path = Path(output_path)
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix)).strip("_")
    return path.with_name(f"{path.stem}_{safe_suffix}{path.suffix}")


def _summary_class_suffix(summary_df):
    class_index = summary_df["class_index"].iloc[0]
    if pd.notna(class_index):
        return f"class{int(class_index)}"
    return summary_df["class_label"].iloc[0]


def plot_attribution_summary(
    summary_df,
    output_path,
    *,
    title=None,
    score_column="attribution",
    figsize=None,
    dpi=180,
):
    """Save a horizontal bar chart for a ranked attribution summary."""
    import matplotlib.pyplot as plt

    if summary_df.empty:
        raise ValueError("summary_df must contain at least one row")
    if score_column not in summary_df.columns:
        raise ValueError(f"score_column must exist in summary_df, got {score_column!r}")

    plot_df = summary_df.sort_values("rank", ascending=False).copy()
    labels = plot_df["label"].astype(str)
    if plot_df["class_label"].nunique() > 1:
        labels = plot_df["class_label"].astype(str) + " / " + labels

    values = plot_df[score_column].astype(float)
    colors = [
        _POSITIVE_COLOR if value >= 0 else _NEGATIVE_COLOR
        for value in plot_df["attribution"].astype(float)
    ]

    if figsize is None:
        figsize = (10, max(3.2, 0.44 * len(plot_df) + 1.4))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#404040", linewidth=0.8)
    max_abs_value = float(values.abs().max())
    if max_abs_value > 0:
        ax.set_xlim(-max_abs_value * 1.08, max_abs_value * 1.08)
    ax.set_xlabel(score_column.replace("_", " "))
    if title:
        ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_summary_outputs(
    summary_df,
    *,
    csv_path=None,
    plot_path=None,
    title=None,
    score_column="attribution",
):
    saved_paths = []

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    if plot_path is not None:
        class_labels = summary_df["class_label"].drop_duplicates().tolist()
        if len(class_labels) <= 1:
            saved_paths.append(
                plot_attribution_summary(
                    summary_df,
                    plot_path,
                    title=title,
                    score_column=score_column,
                )
            )
        else:
            for class_label in class_labels:
                class_summary = summary_df[summary_df["class_label"] == class_label]
                suffix = _summary_class_suffix(class_summary)
                class_path = _suffixed_output_path(plot_path, suffix)
                class_title = f"{title} - {class_label}" if title else str(class_label)
                saved_paths.append(
                    plot_attribution_summary(
                        class_summary,
                        class_path,
                        title=class_title,
                        score_column=score_column,
                    )
                )

    return saved_paths


def _indexed_plot_path(plot_path, index):
    """Add a graph index suffix when saving multiple local explanation plots."""
    path = Path(plot_path)
    return path.with_name(f"{path.stem}_graph{index}{path.suffix}")


def _class_plot_path(plot_path, class_idx):
    """Add a class suffix when saving multiple class-specific plots."""
    path = Path(plot_path)
    return path.with_name(f"{path.stem}_class{class_idx}{path.suffix}")


# ----------------------------------------------------------------------
# 1.  Tensor helpers
# ----------------------------------------------------------------------
def collate_graphs(graphs, device):
    """
    Convert *list of DGLGraphs* → batched tensors.
    Returns (node_feats, adjacency) both on `device`.
      node_feats : (B, N, F)
      adjacency  : (B, N, N)
    """
    feats, adjs = [], []
    for g in graphs:
        feats.append(g.ndata["feature"].float())
        adjs.append(g.adjacency_matrix().to_dense())
    return (
        torch.stack(feats, 0).to(device),
        torch.stack(adjs, 0).to(device),
    )


# ----------------------------------------------------------------------
# 2.  Wrapper so Captum / SHAP see simple tensors
# ----------------------------------------------------------------------
class IGWrapper(torch.nn.Module):
    """
    Makes your GraphTransformer look like:
        logits = f(node_features, adjacency_matrix)
    so Captum & SHAP can treat node_features as the *input* we attribute over.
    """

    def __init__(self, graph_transformer_model: torch.nn.Module, device):
        super().__init__()
        self.gtn = graph_transformer_model.eval()  # freeze BN/dropout for XAI
        self.device = device
        self.num_nodes = self.gtn.number_nodes  # constant N

    def forward(self, node_features, adjacency_matrix):
        """
        node_features   : (B, N, F)  – *attributable input*
        adjacency_matrix: (B, N, N)  – kept constant (extra arg)
        """
        B, N, F = node_features.shape
        assert N == self.num_nodes, "Mismatch in number of nodes"

        # Re-build DGLGraphs on the fly so GraphTransformer can consume them
        graphs = []
        for b in range(B):
            idx = adjacency_matrix[b].nonzero(as_tuple=True)
            g = dgl.graph(idx, num_nodes=N)
            g = dgl.add_self_loop(g)
            g.ndata["feature"] = node_features[b]
            graphs.append(g.to(self.device))

        return self.gtn(graphs)  # → logits  (B, num_classes)


# ----------------------------------------------------------------------
# 2-bis.  SHAP-only wrapper:  input = node_features ; adjacency looked up internally
# ----------------------------------------------------------------------
class SHAPWrapper(torch.nn.Module):
    """
    Exposes *only* node_features to SHAP.  The corresponding adjacency
    matrices are supplied at construction time and automatically repeated
    so that every row of `node_features` gets the right A.
    """

    def __init__(self, gnn, adjacencies, device):
        super().__init__()
        self.gnn = gnn.eval()
        self.A_ref = adjacencies  # (K, N, N)  for K reference graphs
        self.device = device

    def _pick_A(self, B):
        """Repeat / slice stored adjacencies so that batch size matches B."""
        if B == self.A_ref.shape[0]:
            return self.A_ref
        # SHAP often extends the batch by repeating reference samples; do the same for A
        reps = (B + self.A_ref.shape[0] - 1) // self.A_ref.shape[0]
        return self.A_ref.repeat((reps, 1, 1))[:B]

    def forward(self, node_features):
        B, N, _ = node_features.shape
        A_batch = self._pick_A(B)

        graphs = []
        for b in range(B):
            idx = A_batch[b].nonzero(as_tuple=True)
            g = dgl.graph(idx, num_nodes=N).to(self.device)
            g = dgl.add_self_loop(g)
            g.ndata["feature"] = node_features[b]
            graphs.append(g)

        return self.gnn(graphs)  # → logits


# ----------------------------------------------------------------------
# 3.  Integrated Gradients
# ----------------------------------------------------------------------
def explain_with_integrated_gradients(
    model,
    graph,
    target_class=None,
    steps=50,
    attribute_names=None,  # ← list like ["Src Port", "Dst Port", ...]
    plot_path=None,
    plot_title=None,
    class_name=None,
    summary_top_k=10,
    summary_level="node_feature",
    summary_plot_path=None,
    summary_csv_path=None,
):

    # ------------------------------------------------------------------ #
    #  run IG
    # ------------------------------------------------------------------ #
    device = next(model.parameters()).device
    wrapper = IGWrapper(model, device)
    x, A = collate_graphs([graph], device)  # (1, N, F)

    if target_class is None:
        with torch.no_grad():
            target_class = wrapper(x, A).argmax(-1).item()

    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(x)
    attrib, _ = ig.attribute(
        inputs=x,
        baselines=baseline,
        target=target_class,
        additional_forward_args=A,
        n_steps=steps,
        return_convergence_delta=True,
    )

    # ------------------------------------------------------------------ #
    #  formatting
    # ------------------------------------------------------------------ #
    # (1, N, F) -> (N, F)
    mat = attrib.squeeze(0).cpu().numpy()  # N rows, F cols
    N, F = mat.shape

    if attribute_names is None or len(attribute_names) != F:
        attribute_names = [f"feat_{j}" for j in range(F)]

    node_index = [f"Node {i}" for i in range(N)]

    df = pd.DataFrame(mat, index=node_index, columns=attribute_names)

    # ------------------------------------------------------------------ #
    #  pretty-print
    # ------------------------------------------------------------------ #
    print(f"\nIntegrated Gradients (Class {target_class})")
    print(df.round(4))

    if summary_top_k is not None and int(summary_top_k) > 0:
        summary_df = summarize_attribution_df(
            df,
            class_index=target_class,
            class_name=class_name,
            top_k=summary_top_k,
            level=summary_level,
        )
        df.attrs["summary"] = summary_df
        summary_title = f"Integrated Gradients top {summary_top_k} contributors"
        _print_attribution_summary(summary_df, summary_title)
        saved_paths = _save_summary_outputs(
            summary_df,
            csv_path=summary_csv_path,
            plot_path=summary_plot_path,
            title=summary_title,
        )
        for saved_to in saved_paths:
            print(f"\nSaved IG attribution summary to: {saved_to}")

    if plot_path is not None:
        saved_to = plot_quadrant_attributions(
            graph,
            df,
            plot_path,
            title=plot_title or f"Integrated Gradients - Class {target_class}",
        )
        print(f"\nSaved IG quadrant attribution graph to: {saved_to}")

    return df  # return DataFrame for further use


def summarize_integrated_gradients(
    model,
    graphs,
    *,
    target_classes,
    steps=50,
    attribute_names=None,
    class_name_fn=None,
    summary_top_k=10,
    summary_level="node_feature",
    summary_plot_path=None,
    summary_csv_path=None,
):
    """
    Compute an Integrated Gradients top-attribution summary over multiple graphs.

    The signed attribution is averaged over graphs for bar direction, while
    mean absolute attribution is used for ranking important contributors.
    """
    if summary_top_k is None or int(summary_top_k) <= 0:
        return pd.DataFrame()

    device = next(model.parameters()).device
    wrapper = IGWrapper(model, device)
    x, A = collate_graphs(graphs, device)
    B, N, F = x.shape

    if attribute_names is None or len(attribute_names) != F:
        attribute_names = [f"feat_{j}" for j in range(F)]

    node_index = [f"Node {i}" for i in range(N)]
    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(x)

    summary_tables = []
    for class_idx in target_classes:
        attrib = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=int(class_idx),
            additional_forward_args=A,
            n_steps=steps,
        )
        mat = attrib.detach().cpu().numpy()
        signed_df = pd.DataFrame(
            mat.mean(axis=0),
            index=node_index,
            columns=attribute_names,
        )
        abs_df = pd.DataFrame(
            np.abs(mat).mean(axis=0),
            index=node_index,
            columns=attribute_names,
        )
        class_name = class_name_fn(class_idx) if class_name_fn is not None else None
        class_summary = summarize_attribution_df(
            signed_df,
            abs_attribution_df=abs_df,
            class_index=int(class_idx),
            class_name=class_name,
            top_k=summary_top_k,
            level=summary_level,
        )
        class_summary.insert(0, "num_graphs", B)
        summary_tables.append(class_summary)

    summary_df = pd.concat(summary_tables, ignore_index=True)
    summary_title = f"Integrated Gradients global top {summary_top_k} contributors over {B} graphs"
    _print_attribution_summary(summary_df, summary_title)
    saved_paths = _save_summary_outputs(
        summary_df,
        csv_path=summary_csv_path,
        plot_path=summary_plot_path,
        title=summary_title,
    )
    for saved_to in saved_paths:
        print(f"\nSaved IG attribution summary to: {saved_to}")

    return summary_df


# ----------------------------------------------------------------------
# 4.  SHAP (GradientExplainer)
# ----------------------------------------------------------------------
def explain_with_shap(
    model,
    background_graphs,
    sample_graphs,
    *,
    mode="auto",  # "auto" | "global" | "local"
    background_size=50,
    nsamples=32,
    attribute_names=None,
    plot_path=None,
    plot_class=None,
    plot_title_context=None,
    plot_title_builder=None,
    summary_top_k=10,
    summary_level="node_feature",
    summary_plot_path=None,
    summary_csv_path=None,
):

    device = next(model.parameters()).device
    ref_x, ref_A = collate_graphs(background_graphs[:background_size], device)
    samp_x, _ = collate_graphs(sample_graphs, device)  # (B,N,F)

    shap_wrapper = SHAPWrapper(model, ref_A, device)
    explainer = shap.GradientExplainer(shap_wrapper, ref_x)

    shap_vals = explainer.shap_values(samp_x, nsamples=nsamples)  # (B,N,F,C)
    if isinstance(shap_vals, list):
        shap_vals = np.stack(shap_vals, axis=-1)
    B, N, F, C = shap_vals.shape

    # ------------------------------------------------------------------ #
    #  build flattened feature labels  "Node i Feature <attr>"
    # ------------------------------------------------------------------ #
    if attribute_names is None or len(attribute_names) != F:
        plot_attribute_names = [f"feat_{j}" for j in range(F)]
        feature_names = [f"Node {i} Feature feat_{j}" for i in range(N) for j in range(F)]
    else:
        plot_attribute_names = list(attribute_names)
        feature_names = [
            f"Node {i} Feature {plot_attribute_names[j]}"
            for i in range(N)
            for j in range(F)
        ]

    if mode == "auto":
        mode = "local" if B == 1 else "global"

    # ------------------------------------------------------------------ #
    #  GLOBAL  : average |SHAP| over graphs  ➜  classes × flattened feat
    # ------------------------------------------------------------------ #
    if mode == "global":
        per_class_abs = (
            torch.from_numpy(shap_vals).abs().mean(dim=0).numpy()  # over B
        )  # (N,F,C)

        data = {f"Class {c}": per_class_abs[..., c].flatten() for c in range(C)}
        df = pd.DataFrame(data).T
        df.columns = feature_names  # attach proper labels

        print("\nSHAP (Global):")
        print(df.round(4))

        if summary_top_k is not None and int(summary_top_k) > 0:
            per_class_signed = shap_vals.mean(axis=0)  # (N,F,C)
            node_index = [f"Node {i}" for i in range(N)]
            summary_tables = []
            for class_idx in range(C):
                class_name = (
                    plot_title_context(class_idx)
                    if plot_title_context is not None
                    else None
                )
                signed_df = pd.DataFrame(
                    per_class_signed[:, :, class_idx],
                    index=node_index,
                    columns=plot_attribute_names,
                )
                abs_df = pd.DataFrame(
                    per_class_abs[:, :, class_idx],
                    index=node_index,
                    columns=plot_attribute_names,
                )
                summary_tables.append(
                    summarize_attribution_df(
                        signed_df,
                        abs_attribution_df=abs_df,
                        class_index=class_idx,
                        class_name=class_name,
                        top_k=summary_top_k,
                        level=summary_level,
                    )
                )

            summary_df = pd.concat(summary_tables, ignore_index=True)
            df.attrs["summary"] = summary_df
            summary_title = f"SHAP global top {summary_top_k} contributors"
            _print_attribution_summary(summary_df, summary_title)
            saved_paths = _save_summary_outputs(
                summary_df,
                csv_path=summary_csv_path,
                plot_path=summary_plot_path,
                title=summary_title,
                score_column="attribution",
            )
            for saved_to in saved_paths:
                print(f"\nSaved SHAP attribution summary to: {saved_to}")

        if plot_path is not None:
            print("Skipping SHAP quadrant graph for global mode; use mode='local'.")
        return df

    # ------------------------------------------------------------------ #
    #  LOCAL   : one table *per graph*  (classes × flattened feat)
    # ------------------------------------------------------------------ #
    local_tables = []
    local_summary_tables = []
    for b in range(B):
        data = {f"Class {c}": shap_vals[b, :, :, c].flatten() for c in range(C)}
        df = pd.DataFrame(data).T
        df.columns = feature_names  # attach labels

        print(f"\nSHAP (Local) – Graph {b}")
        print(df.round(4))

        if summary_top_k is not None and int(summary_top_k) > 0:
            node_index = [f"Node {i}" for i in range(N)]
            graph_summary_tables = []
            for class_idx in range(C):
                class_name = (
                    plot_title_context(class_idx)
                    if plot_title_context is not None
                    else None
                )
                class_df = pd.DataFrame(
                    shap_vals[b, :, :, class_idx],
                    index=node_index,
                    columns=plot_attribute_names,
                )
                class_summary = summarize_attribution_df(
                    class_df,
                    class_index=class_idx,
                    class_name=class_name,
                    top_k=summary_top_k,
                    level=summary_level,
                )
                class_summary.insert(0, "graph_index", b)
                graph_summary_tables.append(class_summary)

            graph_summary_df = pd.concat(graph_summary_tables, ignore_index=True)
            df.attrs["summary"] = graph_summary_df
            local_summary_tables.append(graph_summary_df)
            summary_title = f"SHAP local top {summary_top_k} contributors - Graph {b}"
            _print_attribution_summary(graph_summary_df, summary_title)

            if summary_plot_path is not None:
                current_summary_plot_path = summary_plot_path
                if B > 1:
                    current_summary_plot_path = _indexed_plot_path(current_summary_plot_path, b)
                saved_paths = _save_summary_outputs(
                    graph_summary_df,
                    plot_path=current_summary_plot_path,
                    title=summary_title,
                )
                for saved_to in saved_paths:
                    print(f"\nSaved SHAP attribution summary to: {saved_to}")

        local_tables.append(df)

        if plot_path is not None:
            if plot_class == "all":
                class_indices = list(range(C))
            elif plot_class is None:
                with torch.no_grad():
                    class_indices = [shap_wrapper(samp_x[b:b + 1]).argmax(-1).item()]
            else:
                class_indices = [int(plot_class)]

            bad_classes = [c for c in class_indices if not 0 <= c < C]
            if bad_classes:
                raise ValueError(
                    f"plot_class must be 'all' or in [0, {C - 1}], got {bad_classes[0]}"
                )

            node_index = [f"Node {i}" for i in range(N)]
            for class_idx in class_indices:
                class_mat = shap_vals[b, :, :, class_idx]
                plot_df = pd.DataFrame(
                    class_mat,
                    index=node_index,
                    columns=plot_attribute_names,
                )
                current_plot_path = plot_path
                if B > 1:
                    current_plot_path = _indexed_plot_path(current_plot_path, b)
                if len(class_indices) > 1:
                    current_plot_path = _class_plot_path(current_plot_path, class_idx)

                saved_to = plot_quadrant_attributions(
                    sample_graphs[b],
                    plot_df,
                    current_plot_path,
                    title=(
                        plot_title_builder(b, class_idx)
                        if plot_title_builder is not None
                        else (
                            f"SHAP - Graph {b} - Class {class_idx}"
                            if plot_title_context is None
                            else f"SHAP - Graph {b} - Attributed: {plot_title_context(class_idx)}"
                        )
                    ),
                )
                print(f"\nSaved SHAP quadrant attribution graph to: {saved_to}")

    if local_summary_tables and summary_csv_path is not None:
        combined_summary_df = pd.concat(local_summary_tables, ignore_index=True)
        saved_paths = _save_summary_outputs(
            combined_summary_df,
            csv_path=summary_csv_path,
        )
        for saved_to in saved_paths:
            print(f"\nSaved SHAP attribution summary to: {saved_to}")

    return local_tables if B > 1 else local_tables[0]


# ----------------------------------------------------------------------
# 6.  Ready-to-use metric helpers  (IG  &  SHAP)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 6.  Metric helpers  (IG  &  SHAP)
# ----------------------------------------------------------------------

# -------- generic utilities -------------------------------------------
def _mask_topk(x, attr, k_frac, *, keep=True):
    """
    Return a copy of `x` where the **top-k %** (by |attr|)
    are either kept (keep=True) or zeroed (keep=False).
    """
    k = int(k_frac * attr.numel())
    if k == 0:
        return x.clone()

    flat = attr.abs().flatten()
    thresh = torch.topk(flat, k).values.min()
    mask = (attr.abs() >= thresh).float()          # 1 = important
    return x * mask if keep else x * (1 - mask)


def _faithfulness_auc(
    forward_fn,
    x,                       # (1,N,F)  – same device as model
    attr,                    # (1,N,F)
    *,
    additional_forward_args=None,
    target=None,
    max_remove_frac=0.20,
    num_points=20,
):
    """
    Deletion‐curve Faithfulness-AUC (lower = more faithful).
    No Captum required.
    """
    device = x.device
    base_logit = forward_fn(
        x, *([] if additional_forward_args is None else additional_forward_args)
    )[0, target].item()

    # rank features
    flat_attr = attr.abs().flatten()
    ranked_idx = torch.argsort(flat_attr, descending=True)

    max_remove = int(max_remove_frac * flat_attr.numel())
    step = max(1, max_remove // num_points)
    fracs, drops = [], []

    x_work = x.clone()  # will be edited in-place
    for k in range(0, max_remove + 1, step):
        if k > 0:
            idx_chunk = ranked_idx[(k - step):k]
            x_work.view(-1)[idx_chunk] = 0.0       # delete next chunk

        with torch.no_grad():
            logit = forward_fn(
                x_work, *([] if additional_forward_args is None else additional_forward_args)
            )[0, target].item()

        fracs.append(k / flat_attr.numel())
        drops.append(base_logit - logit)

    return float(np.trapz(drops, fracs))           # trapezoidal AUC


# ======================================================================
#  IG metrics
# ======================================================================
def compute_ig_metrics(
    model,
    graphs,
    *,
    steps=50,
    noise_std=0.01,
    n_perturb=50,
    k_fracs=(0.05, 0.10),         # for suff / comp
):
    """
    Returns a dict with averages over graphs
        { 'faithfulness_auc': …,
          'sufficiency'     : …,
          'comprehensiveness': …,
          'infidelity'      : …,
          'sensitivity_max' : … }
    """
    device  = next(model.parameters()).device
    wrap    = IGWrapper(model, device)
    ig      = IntegratedGradients(wrap, multiply_by_inputs=False)

    # perturb-func for infidelity
    @infidelity_perturb_func_decorator(multiply_by_inputs=False)
    def _noise(x):
        return x + torch.randn_like(x) * noise_std

    metrics = {m: [] for m in
               ("faithfulness_auc", "sufficiency", "comprehensiveness",
                "infidelity", "sensitivity_max")}

    for g in graphs:
        x, A = collate_graphs([g], device)
        tgt  = wrap(x, A).argmax(-1)

        # --- attribution ------------------------------------------------
        attr = ig.attribute(
            x, baselines=torch.zeros_like(x), target=tgt,
            additional_forward_args=A, n_steps=steps)

        # ---------- Faithfulness-AUC ------------------------------------
        metrics["faithfulness_auc"].append(
            _faithfulness_auc(
                forward_fn=wrap,
                x=x, attr=attr,
                additional_forward_args=(A,),
                target=tgt))

        # ---------- Sufficiency & Comprehensiveness ---------------------
        suff, comp = 0.0, 0.0
        for k in k_fracs:
            keep_top = _mask_topk(x, attr, k, keep=True)
            drop_top = _mask_topk(x, attr, k, keep=False)

            with torch.no_grad():
                base = wrap(x,        A)[0, tgt]
                keep = wrap(keep_top, A)[0, tgt]
                drop = wrap(drop_top, A)[0, tgt]

            suff += (base - keep).abs().item()
            comp += (base - drop).abs().item()

        metrics["sufficiency"].append(suff / len(k_fracs))
        metrics["comprehensiveness"].append(comp / len(k_fracs))

        # ---------- Infidelity -----------------------------------------
        metrics["infidelity"].append(
            infidelity(
                wrap.forward, _noise,
                inputs=x, attributions=attr,
                additional_forward_args=A, target=tgt,
                n_perturb_samples=n_perturb).item())

        # ---------- Sensitivity-max ------------------------------------
        metrics["sensitivity_max"].append(
            sensitivity_max(
                ig.attribute, inputs=x, additional_forward_args=A, target=tgt,
                perturb_radius=noise_std, n_perturb_samples=n_perturb).item())

    # average over graphs
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ======================================================================
#  SHAP metrics   (uses GradientExplainer)
# ======================================================================
def compute_shap_metrics(
    model,
    background_graphs,
    sample_graphs,
    *,
    nsamples=32,
    noise_std=0.01,
    n_perturb=50,
    k_fracs=(0.05, 0.10),
):
    device  = next(model.parameters()).device
    ref_x, ref_A = collate_graphs(background_graphs, device)
    shap_wrap = SHAPWrapper(model, ref_A, device)

    shap_expl = shap.GradientExplainer(shap_wrap, ref_x)

    def shap_attr(x, target):
        sv = shap_expl.shap_values(x.detach().clone().requires_grad_(True),
                               nsamples=nsamples)
        sv_c = sv[target] if isinstance(sv, list) else sv[..., target]
        return torch.from_numpy(sv_c).to(device)      # (1,N,F)

    metrics = {m: [] for m in
               ("faithfulness_auc", "sufficiency", "comprehensiveness",
                "infidelity", "sensitivity_max")}

    for g in sample_graphs:
        x, _ = collate_graphs([g], device)
        tgt  = shap_wrap(x).argmax(-1)

        attr = shap_attr(x, tgt)                      # (1,N,F)

        # ---------- Faithfulness-AUC ------------------------------------
        metrics["faithfulness_auc"].append(
            _faithfulness_auc(
                forward_fn=shap_wrap, x=x, attr=attr, target=tgt))

        # ---------- Sufficiency & Comprehensiveness ---------------------
        suff, comp = 0.0, 0.0
        for k in k_fracs:
            keep_top = _mask_topk(x, attr, k, keep=True)
            drop_top = _mask_topk(x, attr, k, keep=False)

            with torch.no_grad():
                base = shap_wrap(x       )[0, tgt]
                keep = shap_wrap(keep_top)[0, tgt]
                drop = shap_wrap(drop_top)[0, tgt]

            suff += (base - keep).abs().item()
            comp += (base - drop).abs().item()

        metrics["sufficiency"].append(suff / len(k_fracs))
        metrics["comprehensiveness"].append(comp / len(k_fracs))

        # ---------- Infidelity  (custom, no wrapper noise needed) ------
        def _noise(z):                                # perturb-func
            eps = torch.randn_like(z) * noise_std
            return eps, z + eps

        metrics["infidelity"].append(
            infidelity(
                shap_wrap.forward, _noise,
                inputs=x, attributions=attr, target=tgt,
                n_perturb_samples=n_perturb).item())

        # ---------- Sensitivity-max  (simple loop) ---------------------
        base_attr = attr
        max_delta = 0.0
        for _ in range(n_perturb):
            noisy = x + torch.randn_like(x) * noise_std
            delta = (shap_attr(noisy, tgt) - base_attr).abs().max().item()
            max_delta = max(max_delta, delta)
        metrics["sensitivity_max"].append(max_delta)

    return {k: float(np.mean(v)) for k, v in metrics.items()}
