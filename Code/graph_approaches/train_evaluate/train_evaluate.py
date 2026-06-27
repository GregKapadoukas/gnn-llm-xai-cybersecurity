import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from train_evaluate.distributed import (
    DistributedContext,
    is_distributed_available,
    load_checkpoint_state,
)


def train(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer,
    graphs_train,
    labels_train,
    batch_size,
    epoch_num,
    device,
    evaluation_mode,
):
    total_loss = 0.0
    mean_batch_loss = 0.0
    batch_labels_pred = torch.empty(
        0,
    ).to(device)
    batch_labels = torch.empty(
        0,
    ).to(device)
    num_batches = math.ceil(len(graphs_train) / batch_size) - 1

    for batch_num in range(0, num_batches):
        # Split graphs and labels
        graphs, labels = getBatch(graphs_train, labels_train, batch_num, batch_size)

        # Zero optimizer gradients for each batch
        optimizer.zero_grad()

        # Make predictions for batch
        labels_pred = model(graphs)
        batch_labels_pred = torch.cat((batch_labels_pred, labels_pred))
        batch_labels = torch.cat((batch_labels, labels))

        # Compute loss and gradients
        loss = loss_function(labels_pred, labels)
        loss.backward()

        # Adjust model weights
        optimizer.step()

        # Gather data and report
        total_loss += loss.item()
        if batch_num % 10 == 9:
            mean_batch_loss = total_loss / 10  # loss per batch
            batch_labels = torch.argmax(batch_labels, dim=1)
            accuracy = multiclass_accuracy(
                batch_labels_pred,
                batch_labels,
                average="macro",
                num_classes=labels_pred.size()[1],
            )
            if evaluation_mode["mode"] == "cv":
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                    f"| accuracy {accuracy} "
                    f"| fold {evaluation_mode['fold']} "
                )
            elif evaluation_mode["mode"] == "train-test-dev":
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                    f"| accuracy {accuracy} "
                    f"| set {evaluation_mode['set']} "
                )
            else:
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                    f"| accuracy {accuracy} "
                )
            total_loss = 0.0
            batch_labels_pred = torch.empty(
                0,
            ).to(device)
            batch_labels = torch.empty(
                0,
            ).to(device)

        # if batch_num == 100:
        #    break

    return mean_batch_loss


def evaluate(
    model: nn.Module,
    checkpoint_path,
    loss_function: nn.Module,
    graphs_test,
    labels_test,
    discrete_labels,
    batch_size,
    device,
    evaluation_mode,
):
    load_checkpoint_state(model, checkpoint_path, device)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        mean_batch_loss = 0.0
        num_batches = math.ceil(len(graphs_test) / batch_size) - 1

        labels_pred = torch.empty(0, labels_test.size()[1]).to(device)
        # labels_true = torch.empty(
        #    0,
        # ).to(device)

        for batch_num in range(0, num_batches):
            # Split graphs and labels
            graphs, labels = getBatch(graphs_test, labels_test, batch_num, batch_size)

            # Make predictions for batch
            batch_labels_pred = model(graphs)
            labels_pred = torch.cat((labels_pred, batch_labels_pred))

            # labels_true = torch.cat((labels_true, labels))

            # Compute loss and gradients
            loss = loss_function(batch_labels_pred, labels)

            # Gather data and report
            batch_loss = loss.item()
            total_loss += batch_loss

            batch_labels_true = torch.argmax(labels, dim=1)
            accuracy = multiclass_accuracy(
                batch_labels_pred,
                batch_labels_true,
                average="micro",
                num_classes=batch_labels_pred.size()[1],
            )

            if batch_num % 10 == 9:
                if evaluation_mode["mode"] == "cv":
                    print(
                        f"| evaluation batch {batch_num+1}/{num_batches} | batch loss {batch_loss} | accuracy {accuracy} | fold {evaluation_mode['fold']}"
                    )
                if evaluation_mode["mode"] == "train-test-dev":
                    print(
                        f"| evaluation batch {batch_num+1}/{num_batches} | batch loss {batch_loss} | accuracy {accuracy} | set {evaluation_mode['set']}"
                    )
                else:
                    print(
                        f"| evaluation batch {batch_num+1}/{num_batches} | batch loss {batch_loss} | accuracy {accuracy}"
                    )

            # if batch_num == 100:
            #    break

        labels_test = labels_test[0 : labels_pred.size()[0]]
        labels_test = torch.argmax(labels_test, dim=1)
        # labels_true = torch.argmax(labels_true, dim=1)

        mean_batch_loss = total_loss / num_batches  # loss per batch
        accuracy = multiclass_accuracy(
            labels_pred, labels_test, average="micro", num_classes=labels_pred.size()[1]
        )
        precision = multiclass_precision(
            labels_pred, labels_test, average="macro", num_classes=labels_pred.size()[1]
        )
        recall = multiclass_recall(
            labels_pred, labels_test, average="macro", num_classes=labels_pred.size()[1]
        )
        f1_score = multiclass_f1_score(
            labels_pred, labels_test, average="macro", num_classes=labels_pred.size()[1]
        )
        print("=" * 89)
        print(
            f"| Evaluation {len(graphs_test)} samples "
            f"| mean batch loss: {mean_batch_loss}\n"
            f"| Metrics: accuracy: {accuracy} "
            f"| macro precision: {precision}\n"
            f"| macro recall: {recall} "
            f"| macro f1-score: {f1_score}"
        )
        print("=" * 89)
        cr = printClassificationReport(
            labels_pred, labels_test, discrete_labels, evaluation_mode
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cr": cr,
        }


def distributed_evaluate(
    model: nn.Module,
    checkpoint_path,
    loss_function: nn.Module,
    graphs_test,
    labels_test,
    discrete_labels,
    batch_size,
    device,
    evaluation_mode,
    distributed_context=None,
    shard_strategy="round-robin",
):
    context = distributed_context or _distributed_context_from_device(device)
    load_checkpoint_state(model, checkpoint_path, device)
    model.eval()
    if context.enabled:
        dist.barrier()

    local_graphs, local_labels = _shard_for_rank(
        graphs_test,
        labels_test,
        context,
        shard_strategy,
    )

    with torch.no_grad():
        local_total_loss = 0.0
        local_num_batches = 0
        local_labels_pred = torch.empty(0, labels_test.size()[1], device=device)
        local_labels_true = torch.empty(0, dtype=torch.long, device=device)

        for batch_num, (graphs, labels) in enumerate(
            _iter_batches(local_graphs, local_labels, batch_size)
        ):
            graphs = [graph.to(device) for graph in graphs]
            labels = labels.to(device)

            batch_labels_pred = model(graphs)
            local_labels_pred = torch.cat((local_labels_pred, batch_labels_pred))
            local_labels_true = torch.cat(
                (local_labels_true, torch.argmax(labels, dim=1))
            )

            loss = loss_function(batch_labels_pred, labels)
            batch_loss = loss.item()
            local_total_loss += batch_loss
            local_num_batches += 1

            if context.is_main and batch_num % 10 == 9:
                batch_labels_true = torch.argmax(labels, dim=1)
                accuracy = multiclass_accuracy(
                    batch_labels_pred,
                    batch_labels_true,
                    average="micro",
                    num_classes=batch_labels_pred.size()[1],
                )
                print(
                    f"| rank {context.rank} evaluation batch "
                    f"{batch_num+1}/{math.ceil(len(local_graphs) / batch_size)} "
                    f"| batch loss {batch_loss} | accuracy {accuracy}"
                )

        labels_pred = _gather_variable_batch_tensor(local_labels_pred, context)
        labels_true = _gather_variable_batch_tensor(local_labels_true, context)
        total_loss, num_batches = _reduce_loss_counts(
            local_total_loss,
            local_num_batches,
            device,
            context,
        )

        if not context.is_main:
            return None

        if labels_pred.size()[0] == 0:
            raise RuntimeError("No evaluation samples were processed")

        mean_batch_loss = total_loss / max(num_batches, 1)
        accuracy = multiclass_accuracy(
            labels_pred, labels_true, average="micro", num_classes=labels_pred.size()[1]
        )
        precision = multiclass_precision(
            labels_pred, labels_true, average="macro", num_classes=labels_pred.size()[1]
        )
        recall = multiclass_recall(
            labels_pred, labels_true, average="macro", num_classes=labels_pred.size()[1]
        )
        f1_score = multiclass_f1_score(
            labels_pred, labels_true, average="macro", num_classes=labels_pred.size()[1]
        )
        print("=" * 89)
        print(
            f"| Distributed evaluation {len(graphs_test)} samples "
            f"| world size: {context.world_size} "
            f"| mean batch loss: {mean_batch_loss}\n"
            f"| Metrics: accuracy: {accuracy} "
            f"| macro precision: {precision}\n"
            f"| macro recall: {recall} "
            f"| macro f1-score: {f1_score}"
        )
        print("=" * 89)
        cr = printClassificationReport(
            labels_pred, labels_true, discrete_labels, evaluation_mode
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cr": cr,
        }


def distributed_throughput_evaluate(
    model: nn.Module,
    checkpoint_path,
    graphs_test,
    labels_test,
    batch_size,
    device,
    evaluation_mode,
    distributed_context=None,
    shard_strategy="round-robin",
    warmup_batches=2,
    include_transfer=False,
):
    context = distributed_context or _distributed_context_from_device(device)
    load_checkpoint_state(model, checkpoint_path, device)
    model.eval()
    if context.enabled:
        dist.barrier()

    local_graphs, local_labels = _shard_for_rank(
        graphs_test,
        labels_test,
        context,
        shard_strategy,
    )
    graph_batches = [
        graphs for graphs, _ in _iter_batches(local_graphs, local_labels, batch_size)
    ]

    if not include_transfer:
        graph_batches = [
            [graph.to(device) for graph in graphs] for graphs in graph_batches
        ]

    use_cuda = str(device).startswith("cuda") or (
        isinstance(device, torch.device) and device.type == "cuda"
    )

    with torch.no_grad():
        for graphs in graph_batches[:warmup_batches]:
            if include_transfer:
                graphs = [graph.to(device) for graph in graphs]
            model(graphs)

        if use_cuda:
            torch.cuda.synchronize(device)
        if context.enabled:
            dist.barrier()

        timer_start = time.perf_counter()
        for graphs in graph_batches:
            if include_transfer:
                graphs = [graph.to(device) for graph in graphs]
            model(graphs)

        if use_cuda:
            torch.cuda.synchronize(device)
        timer_end = time.perf_counter()

    local_elapsed = timer_end - timer_start
    local_samples = sum(len(graphs) for graphs in graph_batches)
    local_batches = len(graph_batches)

    total_samples, total_batches, max_batches, total_inference_time = (
        _reduce_throughput_stats(
            local_samples,
            local_batches,
            local_elapsed,
            device,
            context,
        )
    )

    if not context.is_main:
        return None

    avg_step_latency = total_inference_time / max(max_batches, 1)
    avg_sample_wall_time = total_inference_time / max(total_samples, 1)
    throughput = total_samples / total_inference_time

    print("=" * 89)
    print(
        f"| Distributed throughput evaluation {total_samples} samples "
        f"| world size: {context.world_size} "
        f"| per-rank batch size: {batch_size}\n"
        f"| shard strategy: {shard_strategy} "
        f"| timed rank batches: {total_batches} "
        f"| max rank batches: {max_batches}\n"
        f"| total inference time: {total_inference_time:.6f}s "
        f"| avg distributed step latency: {avg_step_latency * 1000:.6f} ms/step "
        f"| avg sample wall time: {avg_sample_wall_time * 1000:.6f} ms/sample\n"
        f"| throughput: {throughput:.6f} samples/s"
    )
    print("=" * 89)

    return {
        "samples": total_samples,
        "world_size": context.world_size,
        "batch_size_per_rank": batch_size,
        "shard_strategy": shard_strategy,
        "include_transfer": include_transfer,
        "total_inference_time": total_inference_time,
        "avg_step_latency": avg_step_latency,
        "avg_sample_wall_time": avg_sample_wall_time,
        "throughput": throughput,
        "evaluation_mode": evaluation_mode,
    }


def _distributed_context_from_device(device):
    device = torch.device(device)
    if is_distributed_available():
        return DistributedContext(
            enabled=True,
            rank=dist.get_rank(),
            local_rank=device.index or 0,
            world_size=dist.get_world_size(),
            device=device,
        )
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=device.index or 0,
        world_size=1,
        device=device,
    )


def _iter_batches(graphs, labels, batch_size):
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        yield graphs[start:end], labels[start:end]


def _shard_for_rank(graphs, labels, context, shard_strategy="round-robin"):
    if not context.enabled:
        return graphs, labels

    if shard_strategy == "round-robin":
        indices = list(range(context.rank, len(graphs), context.world_size))
    elif shard_strategy == "contiguous":
        shard_size = math.ceil(len(graphs) / context.world_size)
        start = context.rank * shard_size
        end = min(start + shard_size, len(graphs))
        indices = list(range(start, end))
    else:
        raise ValueError(f"Unknown shard strategy: {shard_strategy}")

    return [graphs[i] for i in indices], labels[indices]


def _gather_variable_batch_tensor(tensor, context):
    tensor = tensor.detach().contiguous()
    if not context.enabled:
        return tensor.to("cpu")

    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(sizes, local_size)

    max_size = max(int(size.item()) for size in sizes)
    if tensor.size(0) < max_size:
        padding_shape = (max_size - tensor.size(0),) + tuple(tensor.shape[1:])
        padding = torch.zeros(
            padding_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding), dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, tensor)

    if not context.is_main:
        return None

    return torch.cat(
        [
            rank_tensor[: int(size.item())].to("cpu")
            for rank_tensor, size in zip(gathered, sizes)
        ],
        dim=0,
    )


def _reduce_loss_counts(total_loss, num_batches, device, context):
    totals = torch.tensor(
        [total_loss, float(num_batches)],
        dtype=torch.float64,
        device=device,
    )
    if context.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[0].item(), int(totals[1].item())


def _reduce_throughput_stats(
    local_samples,
    local_batches,
    local_elapsed,
    device,
    context,
):
    count_totals = torch.tensor(
        [float(local_samples), float(local_batches)],
        dtype=torch.float64,
        device=device,
    )
    max_totals = torch.tensor(
        [float(local_batches), local_elapsed],
        dtype=torch.float64,
        device=device,
    )

    if context.enabled:
        dist.all_reduce(count_totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_totals, op=dist.ReduceOp.MAX)

    return (
        int(count_totals[0].item()),
        int(count_totals[1].item()),
        int(max_totals[0].item()),
        max_totals[1].item(),
    )


def getBatch(graphs, labels, i: int, batch_size):
    seq_len = min(batch_size, len(labels) - 1 - i * batch_size)
    graphs_batch = graphs[i * batch_size : i * batch_size + seq_len]
    labels_batch = labels[i * batch_size : i * batch_size + seq_len]
    return graphs_batch, labels_batch


def printClassificationReport(
    labels_pred, labels_test, discrete_labels, evaluation_mode
):
    plt.rcParams.update({"font.size": 25})
    labels_pred = labels_pred.to("cpu")
    labels_test = labels_test.to("cpu")
    labels_pred = labels_pred.numpy()
    labels_test = labels_test.numpy()
    labels_pred = [discrete_labels[np.argmax(p)] for p in labels_pred]
    labels_test = [discrete_labels[p] for p in labels_test]
    cr = classification_report(labels_test, labels_pred, zero_division=0, digits=8)  # type: ignore
    print(cr)
    cm = confusion_matrix(labels_test, labels_pred, labels=discrete_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=discrete_labels)
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xticklabels(disp.display_labels, fontsize=25, rotation=45)
    ax.set_yticklabels(disp.display_labels, fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    if evaluation_mode["mode"] == "cv":
        plt.title(f"Fold {evaluation_mode['fold']}")
        plt.savefig(
            f"Results/Diagrams/{evaluation_mode['name']}-fold-{evaluation_mode['fold']}-confusion_diagram.png",
            dpi=300,
            bbox_inches="tight",
        )
    if evaluation_mode["mode"] == "train-test-dev":
        plt.title(f"Set {evaluation_mode['set']}")
        plt.savefig(
            f"Results/Diagrams/{evaluation_mode['name']}-{evaluation_mode['set']}-confusion_diagram.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"Results/Diagrams/{evaluation_mode['name']}-confusion_diagram.png",
            dpi=300,
            bbox_inches="tight",
        )
    print("=" * 89)
    return cr


def throughput_evaluate(
    model: nn.Module,
    checkpoint_path,
    graphs_test,
    labels_test,
    batch_size,
    device,
    evaluation_mode,
):
    load_checkpoint_state(model, checkpoint_path, device)
    model.eval()

    use_cuda = str(device).startswith("cuda") or (
        isinstance(device, torch.device) and device.type == "cuda"
    )
    with torch.no_grad():
        num_batches = math.ceil(len(graphs_test) / batch_size) - 1

        graphs_batches = []
        for batch_num in range(0, num_batches):
            # Split graphs and labels
            graphs_temp, labels_temp = getBatch(graphs_test, labels_test, batch_num, batch_size)
            graphs_batches.append(graphs_temp)

        if use_cuda:
            torch.cuda.synchronize(device)

        timer_start = time.perf_counter()
        for graphs in graphs_batches:
            # Make predictions for batch
            batch_labels_pred = model(graphs)

        if use_cuda:
            torch.cuda.synchronize(device)

        timer_end = time.perf_counter()
        total_inference_time = timer_end - timer_start

        total_samples = num_batches * batch_size
        avg_batch_latency = total_inference_time / num_batches
        avg_sample_latency = total_inference_time / total_samples
        throughput = total_samples / total_inference_time

        print("=" * 89)
        print(
            f"| Throughput evaluation {len(graphs_test)} samples "
            f"| avg batch latency: {avg_batch_latency * 1000:.6f} ms/batch "
            f"| avg sample latency: {avg_sample_latency * 1000:.6f} ms/sample\n"
            f"| throughput: {throughput:.6f} samples/s"
        )
        print("=" * 89)

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    size_mb = (param_size + buffer_size) / (1000**2)  # This is really MB, not MiB
    return size_mb
