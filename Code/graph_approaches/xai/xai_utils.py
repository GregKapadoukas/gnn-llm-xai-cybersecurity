# xai_utils.py
"""
XAI helpers for the GraphTransformer model defined in graph_transformer.py
=========================================================================

Functions
---------
collate_graphs(...)                   → tensors ready for IG / SHAP
IGWrapper(nn.Module)                  → makes the model look like f(x, A) → logits
SHAPWrapper(nn.Module)                → exposes node features only to SHAP
explain_with_integrated_gradients(...)→ node-level IG on ONE graph
explain_with_shap(...)                → global/local feature importances
compute_ig_metrics(...)               → IG-based metrics (AUC/suff/comp/…)
compute_shap_metrics(...)             → SHAP-based metrics (AUC/suff/comp/…)

All deletion/retention metrics use *probabilities* for the target class.
"""

from __future__ import annotations

import dgl
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.metrics import (infidelity, infidelity_perturb_func_decorator,
                            sensitivity_max)
from dgl import DGLGraph, DGLHeteroGraph

# ----------------------------------------------------------------------
# 1) Tensor helpers
# ----------------------------------------------------------------------
def collate_graphs(graphs, device):
    """
    Convert *list of DGLGraphs* → batched tensors.
    Returns (node_feats, adjacency) both on `device`.
      node_feats : (B, N, F)
      adjacency  : (B, N, N)
    """
    # Normalize input to a list
    if isinstance(graphs, (DGLGraph, DGLHeteroGraph)):
        graphs = [graphs]
    elif not isinstance(graphs, (list, tuple)):
        raise TypeError(f"Expected a graph or list of graphs, got {type(graphs)}")

    feats, adjs = [], []
    for g in graphs:
        feats.append(g.ndata["feature"].float())
        adjs.append(g.adjacency_matrix().to_dense())
    return (
        torch.stack(feats, 0).to(device),
        torch.stack(adjs, 0).to(device),
    )


# ----------------------------------------------------------------------
# 2) Wrappers so Captum / SHAP see simple tensors
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
# 3) Integrated Gradients (pretty printer)
# ----------------------------------------------------------------------
def explain_with_integrated_gradients(
    model,
    graph,
    target_class=None,
    steps=25,
    attribute_names=None,
):
    device = next(model.parameters()).device
    wrapper = IGWrapper(model, device)
    x, A = collate_graphs([graph], device)  # (1, N, F)

    if target_class is None:
        with torch.no_grad():
            target_class = int(wrapper(x, A).argmax(-1).item())

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

    # (1, N, F) -> (N, F)
    mat = attrib.squeeze(0).cpu().numpy()
    N, F_ = mat.shape
    if attribute_names is None or len(attribute_names) != F_:
        attribute_names = [f"feat_{j}" for j in range(F_)]
    node_index = [f"Node {i}" for i in range(N)]
    df = pd.DataFrame(mat, index=node_index, columns=attribute_names)

    print(f"\nIntegrated Gradients (Class {target_class})")
    print(df.round(4))
    return df


# ----------------------------------------------------------------------
# 4) SHAP (GradientExplainer) pretty printer
# ----------------------------------------------------------------------
def explain_with_shap(
    model,
    background_graphs,
    sample_graphs,
    *,
    mode="auto",  # "auto" | "global" | "local"
    background_size=50,
    nsamples=16,
    attribute_names=None,
):
    device = next(model.parameters()).device
    ref_x, ref_A = collate_graphs(background_graphs[:background_size], device)
    samp_x, _ = collate_graphs(sample_graphs, device)  # (B,N,F)

    shap_wrapper = SHAPWrapper(model, ref_A, device)
    explainer = shap.GradientExplainer(shap_wrapper, ref_x)

    shap_vals = explainer.shap_values(samp_x, nsamples=nsamples)  # (B,N,F,C)
    B, N, F_, C = shap_vals.shape

    if attribute_names is None:
        feature_names = [f"Node {i} · Feature {j}" for i in range(N) for j in range(F_)]
    else:
        feature_names = [
            f"Node {i} Feature {attribute_names[j]}"
            for i in range(N)
            for j in range(len(attribute_names))
        ]

    if mode == "auto":
        mode = "local" if B == 1 else "global"

    if mode == "global":
        per_class = (
            torch.from_numpy(shap_vals).abs().mean(dim=0).numpy()  # over B
        )  # (N,F,C)
        data = {f"Class {c}": per_class[..., c].flatten() for c in range(C)}
        df = pd.DataFrame(data).T
        df.columns = feature_names
        print("\nSHAP (Global):")
        print(df.round(4))
        return df

    local_tables = []
    for b in range(B):
        data = {f"Class {c}": shap_vals[b, :, :, c].flatten() for c in range(C)}
        df = pd.DataFrame(data).T
        df.columns = feature_names
        print(f"\nSHAP (Local) – Graph {b}")
        print(df.round(4))
        local_tables.append(df)

    return local_tables if B > 1 else local_tables[0]


# ----------------------------------------------------------------------
# 5) Metric utilities
# ----------------------------------------------------------------------
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
    mask = (attr.abs() >= thresh).float()  # 1 = important (ties may include >k)
    return x * mask if keep else x * (1 - mask)


def _faithfulness_auc(
    forward_fn,
    x,                       # (1,N,F) – same device as model
    attr,                    # (1,N,F)
    *,
    additional_forward_args=None,
    target=None,
    max_remove_frac=0.20,
    num_points=10,
):
    """
    Deletion-curve Faithfulness-AUC (lower = more faithful).

    Curve: class probability p_t after deleting the top-|attr| features.
    The probability is normalised by the initial value so that the curve
    starts at 1 and the AUC lies in [0, 1] (integrated up to max_remove_frac).
    """
    # --- target as int ---------------------------------------------------
    if isinstance(target, torch.Tensor):
        target = int(target.item())
    elif target is not None:
        target = int(target)

    # ---- probability before any deletion -------------------------------
    with torch.no_grad():
        logits0 = forward_fn(
            x, *([] if additional_forward_args is None else additional_forward_args)
        )[0]                                # shape (C,)
        base_prob = F.softmax(logits0, dim=-1)[target].item()

    if base_prob <= 0:
        return 0.0

    # ---- rank features by |attr| ---------------------------------------
    flat_attr = attr.abs().flatten()
    ranked_idx = torch.argsort(flat_attr, descending=True)

    max_remove = int(max_remove_frac * flat_attr.numel())
    step       = max(1, max_remove // num_points)

    fracs, probs = [0.0], [1.0]  # start: nothing removed

    x_work = x.clone()
    for k in range(step, max_remove + 1, step):
        idx_chunk = ranked_idx[(k - step):k]
        x_work.view(-1)[idx_chunk] = 0.0  # delete next chunk

        with torch.no_grad():
            logits = forward_fn(
                x_work, *([] if additional_forward_args is None else additional_forward_args)
            )[0]
            prob = F.softmax(logits, dim=-1)[target].item() / base_prob

        fracs.append(k / flat_attr.numel())
        probs.append(prob)

    return float(np.trapz(probs, fracs))


# ======================================================================
# 6) IG metrics
# ======================================================================
def compute_ig_metrics(
    model,
    graphs,
    *,
    steps=25,
    noise_std=0.01,
    n_perturb=10,
    k_fracs=(0.05, 0.10),         # for suff / comp
):
    """
    Returns a dict with averages over graphs:
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
        tgt  = int(wrap(x, A).argmax(-1).item())  # <-- fixed

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

        # ---------- Sufficiency & Comprehensiveness (probabilities) -----
        suff, comp = 0.0, 0.0
        with torch.no_grad():
            base_prob = F.softmax(wrap(x, A)[0], dim=-1)[tgt].item()

        for k in k_fracs:
            keep_top = _mask_topk(x, attr, k, keep=True)
            drop_top = _mask_topk(x, attr, k, keep=False)

            with torch.no_grad():
                keep_prob = F.softmax(wrap(keep_top, A)[0], dim=-1)[tgt].item()
                drop_prob = F.softmax(wrap(drop_top, A)[0], dim=-1)[tgt].item()

            suff += (base_prob - keep_prob)
            comp += (base_prob - drop_prob)

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
# 7) SHAP metrics   (uses GradientExplainer)
# ======================================================================
def compute_shap_metrics(
    model,
    background_graphs,
    sample_graphs,
    *,
    nsamples=16,
    noise_std=0.01,
    n_perturb=10,
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
        tgt  = int(shap_wrap(x).argmax(-1).item())  # <-- fixed

        attr = shap_attr(x, tgt)                    # (1,N,F)

        # ---------- Faithfulness-AUC ------------------------------------
        metrics["faithfulness_auc"].append(
            _faithfulness_auc(forward_fn=shap_wrap, x=x, attr=attr, target=tgt))

        # ---------- Sufficiency & Comprehensiveness (probabilities) -----
        suff, comp = 0.0, 0.0
        with torch.no_grad():
            base_prob = F.softmax(shap_wrap(x)[0], dim=-1)[tgt].item()

        for k in k_fracs:
            keep_top = _mask_topk(x, attr, k, keep=True)
            drop_top = _mask_topk(x, attr, k, keep=False)

            with torch.no_grad():
                keep_prob = F.softmax(shap_wrap(keep_top)[0], dim=-1)[tgt].item()
                drop_prob = F.softmax(shap_wrap(drop_top)[0], dim=-1)[tgt].item()

            suff += (base_prob - keep_prob)   # no abs(), matches IG path
            comp += (base_prob - drop_prob)

        metrics["sufficiency"].append(suff / len(k_fracs))
        metrics["comprehensiveness"].append(comp / len(k_fracs))

        # ---------- Infidelity  (custom perturb-func) -------------------
        def _noise(z):
            eps = torch.randn_like(z) * noise_std
            return eps, z + eps

        metrics["infidelity"].append(
            infidelity(
                shap_wrap.forward, _noise,
                inputs=x, attributions=attr, target=tgt,
                n_perturb_samples=n_perturb).item())

        # ---------- Sensitivity-max  (sampling-based) -------------------
        base_attr = attr
        max_delta = 0.0
        for _ in range(n_perturb):
            noisy = x + torch.randn_like(x) * noise_std
            delta = (shap_attr(noisy, tgt) - base_attr).abs().max().item()
            max_delta = max(max_delta, delta)
        metrics["sensitivity_max"].append(max_delta)

    return {k: float(np.mean(v)) for k, v in metrics.items()}
