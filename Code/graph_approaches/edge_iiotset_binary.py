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
dataset_path = "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/Edge-IIoTset dataset/"
label_rules = {"Benign": {}}
benign_list = [
    "Normal traffic/Distance/Distance.pcap",
    "Normal traffic/Flame_Sensor/Flame_Sensor.pcap",
    "Normal traffic/Heart_Rate/Heart_Rate.pcap",
    "Normal traffic/IR_Receiver/IR_Receiver.pcap",
    "Normal traffic/Modbus/Modbus.pcap",
    "Normal traffic/phValue/phValue.pcap",
    "Normal traffic/Soil_Moisture/Soil_Moisture.pcap",
    "Normal traffic/Sound_Sensor/Sound_Sensor.pcap",
    "Normal traffic/Temperature_and_Humidity/Temperature_and_Humidity.pcap",
    "Normal traffic/Water_Level/Water_Level.pcap",
]
pcapsToCSVs(
    dataset_path,
    benign_list,
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Benign",
    5000000,
    label_rules,
    0,
)

pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Backdoor_attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Backdoor",
    5000000,
    {"Backdoor": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    [
        "Attack traffic/DDoS HTTP Flood Attacks.pcap",
        "Attack traffic/DDoS TCP SYN Flood Attacks.pcap",
        "Attack traffic/DDoS UDP Flood Attacks.pcap",
    ],  # "Attack traffic/DDoS ICMP Flood Attacks.pcap" is not included because I only look at UDP and TCP packets
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/DDoS",
    5000000,
    {"DDoS": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/MITM (ARP spoofing + DNS) Attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/MITM",
    5000000,
    {"MITM": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/OS Fingerprinting attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/OS Fingerprinting",
    5000000,
    {"OS Fingerprinting": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Password attacks.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Password Attacks",
    5000000,
    {"Password Attacks": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Port Scanning attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Port Scanning",
    5000000,
    {"Port Scanning": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Ransomware attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Ransomware",
    5000000,
    {"Ransomware": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/SQL injection attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/SQL Injection",
    5000000,
    {"SQL Injection": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Uploading attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Upload Attack",
    5000000,
    {"Upload Attack": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/Vulnerability scanner attack.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Vulnerability Scanner",
    5000000,
    {"Vulnerability Scanner": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["Attack traffic/XSS attacks.pcap"],
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/XSS",
    5000000,
    {"XSS": {}},
    0,
)
# -

num_nodes = 20
csvs_paths = [
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Benign/",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Benign/",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Backdoor",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/DDoS",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/MITM",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/OS Fingerprinting",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Password Attacks",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Port Scanning",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Ransomware",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/SQL Injection",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Upload Attack",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/Vulnerability Scanner",
    "../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/CSVs/Malicious/XSS",
]
graphs_path = f"../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/Graphs/Size {num_nodes}/"
network_ips = [
    "104.16.87.20",
    "133.149.252.77",
    "142.250.200.205",
    "142.250.201.10",
    "153.125.214.15",
    "16.226.184.201",
    "166.153.227.121",
    "172.217.19.35",
    "172.217.19.42",
    "183.223.100.122",
    "190.123.219.128",
    "192.168.0.1",
    "192.168.0.101",
    "192.168.0.128",
    "192.168.0.152",
    "192.168.0.170",
    "192.168.1.1",
    "192.168.1.101",
    "192.168.1.128",
    "192.168.2.1",
    "192.168.2.116",
    "192.168.2.194",
    "192.168.3.1",
    "192.168.3.12",
    "192.168.3.18",
    "192.168.4.1",
    "192.168.4.30",
    "192.168.4.73",
    "192.168.5.1",
    "192.168.5.46",
    "192.168.5.47",
    "192.168.6.1",
    "192.168.6.100",
    "192.168.6.56",
    "192.168.7.1",
    "192.168.7.55",
    "192.168.7.62",
    "192.168.8.1",
    "192.168.8.104",
    "192.168.8.163",
    "207.192.25.133",
    "213.117.18.213",
    "216.58.198.74",
    "220.146.94.148",
    "227.117.33.125",
    "49.81.59.152",
    "91.184.12.91",
    "94.196.109.185",
]
loadCSVsAndCreateGraphs(csvs_paths, graphs_path, num_nodes, 10000, "Generalized", "all")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_nodes = 20
epochs = 3
batch_size = 512
number_eigenvectors = 40
embedding_size = 80

# +
one_hot_mapping = {
    "Benign": 1,
    "Backdoor": 2,
    "DDoS": 2,
    "MITM": 2,
    "OS Fingerprinting": 2,
    "Password Attacks": 2,
    "Port Scanning": 2,
    "Ransomware": 2,
    "SQL Injection": 2,
    "Upload Attack": 2,
    "Vulnerability Scanner": 2,
    "XSS": 2,
}
graphs_path = f"../../Datasets/Edge-IIoTset Cyber Security Dataset of IoT & IIoT/My Preprocessing/Graphs/Size {number_nodes}/"
(graphs, labels) = loadGraphDataset(graphs_path, one_hot_mapping, "all")
(
    graphs_train,
    graphs_dev,
    graphs_test,
    labels_train,
    labels_dev,
    labels_test,
) = splitGraphDataset(graphs, labels, 0.1, 0.1, True, device)
# displayGraph(graphs_train[0])

attack_detection_best_model_params_path = os.path.join(
    "../../Checkpoints/",
    f"edge-iiotset-binary-{number_nodes}-{number_eigenvectors}-{embedding_size}.pt",
)

# +
# Train attack detection model
print("Training attack detection model")

evaluation_mode = {
    "mode": "train-test-dev",
    "set": "train",
    "name": "edge-iiotset-binary",
}

attack_detection_model = GraphTransformer(
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
optimizer = torch.optim.Adam(attack_detection_model.parameters(), lr=0.001)
best_train_loss = float("inf")
train_loss = float("inf")

for epoch_num in range(1, epochs + 1):
    graphs_train, labels_train = randomizeGraphOrder(graphs_train, labels_train)
    epoch_start_time = time.time()
    # print(torch.cat((labels_train[:, :1], torch.flip(labels_train[:, :1], [1])), dim=1))
    train_loss = train(
        attack_detection_model,
        loss_function,
        optimizer,
        graphs_train,
        torch.cat((labels_train[:, :1], 1 - labels_train[:, :1]), dim=1),
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
            attack_detection_model.state_dict(),
            attack_detection_best_model_params_path,
        )
print("=" * 89)
print("| Saving final checkpoint")
print("=" * 89)
torch.save(
    attack_detection_model.state_dict(),
    attack_detection_best_model_params_path,
)

evaluation_mode["set"] = "dev"
print("Evaluating attack detection model on dev set")
results = evaluate(
    attack_detection_model,
    attack_detection_best_model_params_path,
    loss_function,
    graphs_dev,
    torch.cat((labels_dev[:, :1], 1 - labels_dev[:, :1]), dim=1),
    ["Benign", "Malicious"],
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/edge-iiotset-binary-results-dev.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

evaluation_mode["set"] = "test"
print("Evaluating attack detection model on test set")
results = evaluate(
    attack_detection_model,
    attack_detection_best_model_params_path,
    loss_function,
    graphs_test,
    torch.cat((labels_test[:, :1], 1 - labels_test[:, :1]), dim=1),
    ["Benign", "Malicious"],
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/edge-iiotset-binary-results-test.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

# Show dev set results
print("=" * 89)
print("Dev set metrics")
print("=" * 89)
with open("Results/Pickle/edge-iiotset-binary-results-dev.pkl", "rb") as file:
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
with open("Results/Pickle/edge-iiotset-binary-results-test.pkl", "rb") as file:
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
