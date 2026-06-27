# +
import os
import pickle
import time
from datetime import datetime
from parser.parser import pcapsToCSVs

import torch
from gnn.graph_transformer import GraphTransformer
from graph_dataset.display_graph import displayGraph
from graph_dataset.graph_dataset import (
    loadGraphDataset,
    oversampleInfrequentClasses,
    randomizeGraphOrder,
    splitGraphDataset,
)
from preprocessing.preprocessor import loadCSVsAndCreateGraphs
from sklearn.model_selection import StratifiedKFold
from torch import nn
from train_evaluate.train_evaluate import evaluate, train

# +
# Looking at the packets, many of the 'Attack times' in the paper are visibly incorrect
# (eg. Training Set NTP contains LDAP packages?  Port 636 is LDAP, not NTP and this is
# clearly script traffic, also SYNs are delayed 14:29 and after)
# So I looked at the packets myself in order to verify the 'Attack times' I use below
# These new 'Attack times' are more accurate than the dataset author's.
# Also as seen in 'Labeling proof.ipynb', I prove that all attacks come from 172.16.0.5,
# since that perfectly splits authors CSVs into benign and malicious packets
# Also, I subtract 3 hours from the timestamps, since they are in UTC and the authors
# seem to be in a UTC -3 timezone

label_rules = {
    "NTP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1023],
    },
    "DNS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [53],
    },
    "LDAP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [636],
    },
    "MSSQL": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1434],
    },
    "NetBIOS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [137],
    },
    "SNMP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [161, 162],
    },
    "SSDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [2869, 5000],
    },
    "UDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 12:45:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:09:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "UDP-Lag": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 13:13:17", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime(
            "2018-12-01 13:26:00", "%Y-%m-%d %H:%M:%S"
        ),  # From 13:11 to 13:13 there are still UDP flood packets. From 13:15 to 13:26 the same attack patters are seen
        "protocol": ["UDP"],
    },
    "WebDDoS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 13:18:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 14:29:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
        "destination_port": [80],
    },
    "SYN": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 14:30:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 17:15:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
    },
    "TFTP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime(
            "2018-12-01 14:40:00", "%Y-%m-%d %H:%M:%S"
        ),  # From 13:35:00 to 14:40:00 and after 15:30:30, the packet sizes are off and no traffic on port 69 so I don't know what the packets are
        "end_time": datetime.strptime("2018-12-01 15:30:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "skip": {
        "ip": ["172.16.0.5"],
    },
    "Benign": {},
}
pcaps_path = "../../Datasets/CIC-DDOS2019/PCAPs/01-12/PCAP-01-12/"
pcaps_name = "SAT-01-12-2018_0"
pcaps_list = []
for i in range(0, 818):
    if i == 0:
        pcaps_list.append(pcaps_name)
    else:
        pcaps_list.append(pcaps_name + str(i))
pcapsToCSVs(
    pcaps_path,
    pcaps_list,
    "../../Datasets/CIC-DDOS2019/My Preprocessing/CSVs/01-12/",
    5000000,
    label_rules,
    3,
)
# -

label_rules = {
    "PortMap": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 09:43:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 09:51:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP", "TCP"],
    },
    "NetBIOS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:00:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:09:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [137],
    },
    "LDAP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:21:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:30:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [636],
    },
    "MSSQL": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:33:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:42:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1434],
    },
    "UDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:53:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 11:03:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "UDP-Lag": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 11:14:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 11:24:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "SYN": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 11:28:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 17:35:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
    },
    "skip": {
        "ip": ["172.16.0.5"],
    },
    "Benign": {},
}
pcaps_path = "../../Datasets/CIC-DDOS2019/PCAPs/03-11/PCAP-03-11/"
pcaps_name = "SAT-03-11-2018_0"
pcaps_list = []
for i in range(0, 146):
    if i == 0:
        pcaps_list.append(pcaps_name)
    else:
        pcaps_list.append(pcaps_name + str(i))
pcapsToCSVs(
    pcaps_path,
    pcaps_list,
    "../../Datasets/CIC-DDOS2019/My Preprocessing/CSVs/03-11/",
    5000000,
    label_rules,
    3,
)

num_nodes = 20
csvs_paths = [
    "../../Datasets/CIC-DDOS2019/My Preprocessing/CSVs/01-12/",
    "../../Datasets/CIC-DDOS2019/My Preprocessing/CSVs/03-11/",
]
graphs_path = f"../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs/Size {num_nodes}/"
loadCSVsAndCreateGraphs(csvs_paths, graphs_path, num_nodes, 10000, "Endpoint", "all")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_nodes = 20
epochs = 3
batch_size = 512
number_eigenvectors = 40 * 2
embedding_size = 80 * 2

# +
# Without the usage of infrequent classes. Using the infrequent classes doesn't make sense since the total amount of samples for the classes is negligible
one_hot_mapping = {
    "Benign": 1,
    "SYN": 2,
    "TFTP": 3,
    "UDP": 4,
    "UDP-Lag": 5,
}
graphs_path = (
    f"../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs/Size {number_nodes}/"
)
(graphs, labels) = loadGraphDataset(graphs_path, one_hot_mapping, 50)
(
    graphs_train,
    graphs_dev,
    graphs_test,
    labels_train,
    labels_dev,
    labels_test,
) = splitGraphDataset(graphs, labels, 0.1, 0.1, True, device)
# displayGraph(graphs_train[0])

classification_best_model_params_path = os.path.join(
    "../../Checkpoints/",
    f"cicddos2019-multiclass-{number_nodes}-{number_eigenvectors}-{embedding_size}.pt",
)

# +
# Train DDoS classification model
print("Training DDoS classification model")

evaluation_mode = {
    "mode": "train-test-dev",
    "set": "train",
    "name": "cicddos2019-multiclass",
}

classification_model = GraphTransformer(
    number_nodes=number_nodes,
    node_features_size=4,
    number_eigenvectors=number_eigenvectors,
    embedding_size=embedding_size,
    feedforward_scaling=20 * 2,
    num_heads=10 * 2,
    num_layers=4,
    dropout=0.5,
    num_classes=5,
    device=device,
).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)
best_train_loss = float("inf")
train_loss = float("inf")

for epoch_num in range(1, epochs + 1):
    graphs_train, labels_train = randomizeGraphOrder(graphs_train, labels_train)
    epoch_start_time = time.time()
    train_loss = train(
        classification_model,
        loss_function,
        optimizer,
        graphs_train,
        labels_train,
        batch_size,
        epoch_num,
        device,
        evaluation_mode,
    )
    elapsed = time.time() - epoch_start_time
    print("-" * 89)
    print(
        f"| end of epoch {epoch_num:3d} | epoch last loss {train_loss} | time: {elapsed:5.2f}s"
    )
    print("-" * 89)

    if train_loss < best_train_loss:
        print("=" * 89)
        print("| Saving new best checkpoint")
        print("=" * 89)
        best_train_loss = train_loss
        torch.save(
            classification_model.state_dict(),
            classification_best_model_params_path,
        )
print("=" * 89)
print("| Saving final checkpoint")
print("=" * 89)
torch.save(
    classification_model.state_dict(),
    classification_best_model_params_path,
)

evaluation_mode["set"] = "dev"
print("Evaluating DDoS detection model with dev set")
results = evaluate(
    classification_model,
    classification_best_model_params_path,
    loss_function,
    graphs_dev,
    labels_dev,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/cicddos2019-multiclass-results-dev.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

evaluation_mode["set"] = "test"
print("Evaluating DDoS detection model with test set")
results = evaluate(
    classification_model,
    classification_best_model_params_path,
    loss_function,
    graphs_test,
    labels_test,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/cicddos2019-multiclass-results-test.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

# Show dev set results
print("=" * 89)
print("Dev set metrics")
print("=" * 89)
with open("Results/Pickle/cicddos2019-multiclass-results-dev.pkl", "rb") as file:
    results = pickle.load(file)
print(
    f"| accuracy: {results['accuracy']} "
    f"| macro precision: {results['precision']}\n"
    f"| macro recall: {results['recall']} "
    f"| macro f1-score: {results['f1_score']}"
)
print("=" * 89)
print("Classification Report")
print(results["cr"])
print("=" * 89)

# Show test set results
print("=" * 89)
print("Test set metrics")
print("=" * 89)
with open("Results/Pickle/cicddos2019-multiclass-results-test.pkl", "rb") as file:
    results = pickle.load(file)
print(
    f"| accuracy: {results['accuracy']} "
    f"| macro precision: {results['precision']}\n"
    f"| macro recall: {results['recall']} "
    f"| macro f1-score: {results['f1_score']}"
)
print("=" * 89)
print("Classification Report")
print(results["cr"])
print("=" * 89)
