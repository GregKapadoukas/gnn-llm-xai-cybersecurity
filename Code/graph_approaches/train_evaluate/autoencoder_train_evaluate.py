import math

import torch
from torch import nn
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from train_evaluate.train_evaluate import getBatch, printClassificationReport


def autoencoder_train(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer,
    graphs_train,
    batch_size,
    epoch_num,
    device,
    evaluation_mode,
):
    total_loss = 0.0
    mean_batch_loss = 0.0
    batch_embeddings_pred = torch.empty(
        0,
    ).to(device)
    batch_embeddings = torch.empty(
        0,
    ).to(device)
    num_batches = math.ceil(len(graphs_train) / batch_size) - 1

    for batch_num in range(0, num_batches):
        # Split graphs
        graphs = getGraphBatch(graphs_train, batch_num, batch_size)

        # Zero optimizer gradients for each batch
        optimizer.zero_grad()

        # Make predictions for batch
        _, embeddings_pred, embeddings_real = model(graphs)
        batch_embeddings_pred = torch.cat((batch_embeddings_pred, embeddings_pred))
        batch_embeddings = torch.cat((batch_embeddings, embeddings_real))

        # Compute loss and gradients
        loss = loss_function(embeddings_pred, embeddings_real)
        loss.backward()

        # Adjust model weights
        optimizer.step()

        # Gather data and report
        total_loss += loss.item()
        if batch_num % 10 == 9:
            mean_batch_loss = total_loss / 10  # loss per batch
            batch_embeddings = torch.argmax(batch_embeddings, dim=1)
            if evaluation_mode["mode"] == "cv":
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                    f"| fold {evaluation_mode['fold']} "
                )
            if evaluation_mode["mode"] == "train-test-dev":
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                    f"| set {evaluation_mode['set']} "
                )
            else:
                print(
                    f"| batch {batch_num+1}/{num_batches} "
                    f"| epoch {epoch_num} "
                    f"| mean batch loss: {mean_batch_loss} "
                )
            total_loss = 0.0
            batch_embeddings_pred = torch.empty(
                0,
            ).to(device)
            batch_embeddings = torch.empty(
                0,
            ).to(device)

        # if batch_num == 100:
        #    break

    return mean_batch_loss


def classifier_train(
    classifier_dnn_model: nn.Module,
    graph_autoencoder_model: nn.Module,
    graph_autoencoder_model_checkpoint_path,
    loss_function: nn.Module,
    optimizer,
    graphs_train,
    labels_train,
    batch_size,
    epoch_num,
    device,
    evaluation_mode,
):
    graph_autoencoder_model.eval()
    graph_autoencoder_model.load_state_dict(
        torch.load(graph_autoencoder_model_checkpoint_path)
    )

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
        embeddings, _, _ = graph_autoencoder_model(graphs)
        labels_pred = classifier_dnn_model(embeddings)
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
            if evaluation_mode["mode"] == "train-test-dev":
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


def autoencoder_dnn_evaluate(
    graph_autoencoder_model: nn.Module,
    classifier_dnn_model: nn.Module,
    graph_autoencoder_model_checkpoint_path,
    classifier_dnn_model_checkpoint_path,
    loss_function,
    graphs_test,
    labels_test,
    discrete_labels,
    batch_size,
    device,
    evaluation_mode,
):
    graph_autoencoder_model.load_state_dict(
        torch.load(graph_autoencoder_model_checkpoint_path)
    )
    graph_autoencoder_model.eval()
    classifier_dnn_model.load_state_dict(
        torch.load(classifier_dnn_model_checkpoint_path)
    )
    classifier_dnn_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        mean_batch_loss = 0.0
        num_batches = math.ceil(len(graphs_test) / batch_size) - 1

        labels_pred = torch.empty(0, labels_test.size()[1]).to(device)

        for batch_num in range(0, num_batches):
            # Split graphs and labels
            graphs, labels = getBatch(graphs_test, labels_test, batch_num, batch_size)

            # Make predictions for batch
            embeddings, _, _ = graph_autoencoder_model(graphs)
            batch_labels_pred = classifier_dnn_model(embeddings)
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


def getGraphBatch(graphs, i: int, batch_size):
    seq_len = min(batch_size, len(graphs) - 1 - i * batch_size)
    graphs_batch = graphs[i * batch_size : i * batch_size + seq_len]
    return graphs_batch
