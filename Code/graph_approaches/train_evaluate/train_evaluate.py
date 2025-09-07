import math

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    model.load_state_dict(torch.load(checkpoint_path))
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
