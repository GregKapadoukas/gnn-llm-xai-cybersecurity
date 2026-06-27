# +
import os
import pickle
import time
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
label_rules = {"Benign": {}}
dataset_path = "../../Datasets/IOT-23/iot_23_datasets_full/opt/Malware-Project/BigDataset/IoTScenarios/"
benign_list = [
    "CTU-Honeypot-Capture-4-1/2018-09-14-13-40-25-Philips-Hue-Bridge.pcap",
    "CTU-Honeypot-Capture-4-1/2018-10-25-14-06-32-192.168.1.132.pcap",
    "CTU-Honeypot-Capture-5-1/2018-09-21-capture.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-01/2019-07-03-15-15-47-first_start_somfy_gateway.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-02/2019-07-03-16-41-09-192.168.1.158.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-03/2019-07-04-16-41-10-192.168.1.158.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-04/2019-07-05-16-41-14-192.168.1.158.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-05/2019-07-06-16-41-17-192.168.1.158.pcap",
    "CTU-Honeypot-Capture-7-1/Somfy-06/2019-07-07-16-41-19-192.168.1.158.pcap",
]
pcapsToCSVs(
    dataset_path,
    benign_list,
    "../../Datasets/IOT-23/My Preprocessing/CSVs/Benign/",
    5000000,
    label_rules,
    0,
)

label_rules = {"Malicious": {}}
malicious_list = [
    "CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap",
    "CTU-IoT-Malware-Capture-3-1/2018-05-21_capture.pcap",
    "CTU-IoT-Malware-Capture-7-1/2018-07-20-17-31-20-192.168.100.108.pcap",
    "CTU-IoT-Malware-Capture-8-1/2018-07-31-15-15-09-192.168.100.113.pcap",
    "CTU-IoT-Malware-Capture-9-1/2018-07-25-10-53-16-192.168.100.111.pcap",
    "CTU-IoT-Malware-Capture-17-1/2018-09-06-11-43-12-192.168.100.111.pcap",
    "CTU-IoT-Malware-Capture-20-1/2018-10-02-13-12-30-192.168.100.103.pcap",
    "CTU-IoT-Malware-Capture-21-1/2018-10-03-15-22-32-192.168.100.113.pcap",
    "CTU-IoT-Malware-Capture-33-1/2018-12-20-21-10-00-192.168.1.197.pcap",
    "CTU-IoT-Malware-Capture-34-1/2018-12-21-15-50-14-192.168.1.195.pcap",
    "CTU-IoT-Malware-Capture-35-1/2018-12-21-15-33-59-192.168.1.196.pcap",
    "CTU-IoT-Malware-Capture-36-1/2018-12-21-13-36-41-192.168.1.198.pcap",
    "CTU-IoT-Malware-Capture-39-1/2019-01-09-21-25-11-192.168.1.194.pcap",
    "CTU-IoT-Malware-Capture-42-1/2019-01-10-14-34-38-192.168.1.197.pcap",
    "CTU-IoT-Malware-Capture-43-1/2019-01-10-19-22-51-192.168.1.198.pcap",
    "CTU-IoT-Malware-Capture-44-1/2019-01-10-21-06-26-192.168.1.199.pcap",
    "CTU-IoT-Malware-Capture-48-1/2019-02-28-19-15-13-192.168.1.200.pcap",
    "CTU-IoT-Malware-Capture-49-1/2019-02-28-20-50-15-192.168.1.193.pcap",
    "CTU-IoT-Malware-Capture-52-1/2019-03-08-13-24-30-192.168.1.197.pcap",
    "CTU-IoT-Malware-Capture-60-1/2019-09-20-02-40-32-192.168.1.195.pcap",
]
pcapsToCSVs(
    dataset_path,
    malicious_list,
    "../../Datasets/IOT-23/My Preprocessing/CSVs/Malicious/",
    5000000,
    label_rules,
    0,
)
# -

num_nodes = 20
csvs_paths = [
    "../../Datasets/IOT-23/My Preprocessing/CSVs_sample/Benign/",
    "../../Datasets/IOT-23/My Preprocessing/CSVs_sample/Malicious/",
]
graphs_path = f"../../Datasets/IOT-23/My Preprocessing/Graphs/Size {num_nodes}/"
network_ips = [
    "192.168.1.1",
    "192.168.1.132",
    "192.168.1.153",
    "192.168.1.193",
    "192.168.1.194",
    "192.168.1.195",
    "192.168.1.197",
    "192.168.1.197",
    "192.168.1.198",
    "192.168.1.199",
    "192.168.1.2",
    "192.168.1.200",
    "192.168.100.1",
    "192.168.100.103",
    "192.168.100.103",
    "192.168.100.108",
    "192.168.100.111",
    "192.168.100.113",
    "192.168.2.1",
    "192.168.2.3",
    "192.168.2.5",
]
loadCSVsAndCreateGraphs(
    csvs_paths, graphs_path, num_nodes, 1000, "Generalized", network_ips
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_nodes = 20
epochs = 10
batch_size = 512
number_eigenvectors = 40
embedding_size = 80

# +
one_hot_mapping = {
    "Benign": 1,
    "Malicious": 2,
}
graphs_path = f"../../Datasets/IOT-23/My Preprocessing/Graphs/Size {number_nodes}/"
(
    graphs,
    labels,
) = loadGraphDataset(graphs_path, one_hot_mapping, 50)
(
    graphs_train,
    graphs_dev,
    graphs_test,
    labels_train,
    labels_dev,
    labels_test,
) = splitGraphDataset(graphs, labels, 0.1, 0.1, True, device)
# displayGraph(graphs_train[0])

botnet_best_model_params_path = os.path.join(
    "../../Checkpoints/",
    f"iot-23-binary-{number_nodes}-{number_eigenvectors}-{embedding_size}.pt",
)
# +
# Train botnet detection model
print("Training botnet detection model")

evaluation_mode = {
    "mode": "train-test-dev",
    "set": "train",
    "name": "iot-23-binary",
}

botnet_model = GraphTransformer(
    number_nodes=number_nodes,
    node_features_size=4,
    number_eigenvectors=number_eigenvectors,
    embedding_size=embedding_size,
    feedforward_scaling=20,
    num_heads=10,
    num_layers=4,
    dropout=0.5,
    num_classes=2,
    device=device,
).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(botnet_model.parameters(), lr=0.001)
best_train_loss = float("inf")
train_loss = float("inf")

for epoch_num in range(1, epochs + 1):
    graphs_train, labels_train = randomizeGraphOrder(graphs_train, labels_train)
    epoch_start_time = time.time()
    # print(torch.cat((labels_train[:, :1], torch.flip(labels_train[:, :1], [1])), dim=1))
    train_loss = train(
        botnet_model,
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
            botnet_model.state_dict(),
            botnet_best_model_params_path,
        )
print("=" * 89)
print("| Saving final checkpoint")
print("=" * 89)
torch.save(
    botnet_model.state_dict(),
    botnet_best_model_params_path,
)

# +
evaluation_mode["set"] = "dev"
print("Evaluating botnet detection model on dev set")
results = evaluate(
    botnet_model,
    botnet_best_model_params_path,
    loss_function,
    graphs_dev,
    labels_dev,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/iot-23-binary-results-dev.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

evaluation_mode["set"] = "test"
print("Evaluating botnet detection model on test set")
results = evaluate(
    botnet_model,
    botnet_best_model_params_path,
    loss_function,
    graphs_test,
    labels_test,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/iot-23-binary-results-test.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

# Show dev set results
print("=" * 89)
print("Dev set metrics")
print("=" * 89)
with open("Results/Pickle/iot-23-binary-results-dev.pkl", "rb") as file:
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
with open("Results/Pickle/iot-23-binary-results-test.pkl", "rb") as file:
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
