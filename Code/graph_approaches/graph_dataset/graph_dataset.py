import math
import os
import pickle

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from natsort import natsorted
from sklearn.model_selection import train_test_split


def loadGraphDataset(graphs_path, one_hot_mapping, files_per_class):
    graphs_files = os.listdir(graphs_path)
    graphs = []
    labels = []
    graphs_files = sampleDataset(graphs_files, files_per_class, one_hot_mapping)
    for file in graphs_files:
        temp_graphs, _ = dgl.load_graphs(graphs_path + file)
        temp_len = len(temp_graphs)
        graphs += temp_graphs[:temp_len]
        for _ in range(temp_len):
            labels.append(
                oneHotEncode(file.split(".")[0].split("_")[1], "cpu", one_hot_mapping)
            )
    labels = torch.stack(labels, dim=0)
    graphs, labels = randomizeGraphOrder(graphs, labels)
    return graphs, labels


def loadDynamicGraphDataset(dynamic_graphs_path, one_hot_mapping, files_per_class):
    graphs_files = os.listdir(dynamic_graphs_path)
    graphs = []
    labels = []
    graphs_files = sampleDataset(graphs_files, files_per_class, one_hot_mapping)
    for file in graphs_files:
        with open(os.path.join(dynamic_graphs_path, file), "rb") as f:
            temp_graphs = pickle.load(f)
        temp_len = len(temp_graphs)
        graphs += temp_graphs[:temp_len]
        for _ in range(temp_len):
            labels.append(
                oneHotEncode(file.split(".")[0].split("_")[1], "cpu", one_hot_mapping)
            )
    labels = torch.stack(labels, dim=0)
    graphs, labels = randomizeGraphOrder(graphs, labels)
    return graphs, labels


def splitGraphDataset(
    graphs, labels, test_size, dev_size, correct_class_imbalance, device
):
    graphs_train, graphs_test, labels_train, labels_test = train_test_split(
        graphs,
        labels,
        random_state=100,
        test_size=test_size,
        shuffle=True,
        stratify=labels,
    )
    graphs_train, graphs_dev, labels_train, labels_dev = train_test_split(
        graphs_train,
        labels_train,
        random_state=100,
        test_size=dev_size / (1 - test_size),
        shuffle=True,
        stratify=labels_train,
    )
    if correct_class_imbalance:
        graphs_train, labels_train = oversampleInfrequentClasses(
            graphs_train, labels_train
        )
    graphs_train = [g.to(device) for g in graphs_train]
    graphs_dev = [g.to(device) for g in graphs_dev]
    graphs_test = [g.to(device) for g in graphs_test]
    labels_train = labels_train.to(device)  # type: ignore
    labels_dev = labels_dev.to(device)  # type: ignore
    labels_test = labels_test.to(device)  # type: ignore
    return graphs_train, graphs_dev, graphs_test, labels_train, labels_dev, labels_test


def oversampleInfrequentClasses(graphs_train, labels_train):
    # Oversample the infrequent classes in the train set to remove class imbalance
    num_classes = labels_train.shape[1]
    ros = RandomOverSampler(random_state=101)
    graphs_train, labels_train = ros.fit_resample(  # type: ignore
        np.array(graphs_train).reshape(-1, 1), torch.argmax(labels_train, dim=1).int().numpy()  # type: ignore
    )
    graphs_train = graphs_train.reshape(-1).tolist()
    labels_train = torch.from_numpy(labels_train)
    labels_train = labels_train.to(torch.int64)
    labels_train = F.one_hot(labels_train.squeeze(-1), num_classes=num_classes).to(
        torch.float
    )
    return graphs_train, labels_train


def randomizeGraphOrder(graphs, labels):
    torch.manual_seed(101)
    perm = torch.randperm(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    labels = torch.stack(labels)
    return graphs, labels


def filterRelevantGraphs(graphs, labels, startBit, endBit, device):
    filtered_graphs = []
    filtered_labels = []
    for i in range(len(graphs)):
        if torch.any(
            labels[i, startBit:endBit] != torch.zeros(endBit - startBit).to(device)
        ):
            filtered_graphs.append(graphs[i])
            filtered_labels.append(labels[i])
    filtered_labels = torch.stack(filtered_labels, dim=0)
    return filtered_graphs, filtered_labels


def oneHotEncode(label, device, one_hot_mapping):
    unique_values = set(value for value in one_hot_mapping.values())
    one_hot_vector = torch.zeros(len(unique_values)).to(device)
    one_hot_vector[one_hot_mapping[label] - 1] = 1
    return one_hot_vector


def sampleDataset(files, files_per_class, one_hot_mapping):
    if files_per_class == "all":
        return files
    attack_classes = {key: {} for key in one_hot_mapping.keys()}
    for file in files:
        try:
            parts = file.split("_")
            file_class = parts[1].split(".")[0]
            file_number = parts[0]
            if file_class in attack_classes.keys():
                attack_classes[file_class][file_number] = file
        except Exception:
            print("Some files were not valid")
    keep = []
    for attack_class in attack_classes.items():
        sorted_numbers = natsorted(attack_class[1].keys())
        interval = len(sorted_numbers) / files_per_class
        for i in range(files_per_class):
            if attack_class[1][sorted_numbers[math.floor(i * interval)]] not in keep:
                keep.append(attack_class[1][sorted_numbers[math.floor(i * interval)]])
    return keep
