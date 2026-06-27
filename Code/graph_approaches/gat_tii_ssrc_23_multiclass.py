# +
import os
import pickle
import time
from parser.parser import pcapsToCSVs

import torch
from gnn.gat import GAT
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
from train_evaluate.train_evaluate import evaluate, train, get_model_size_mb

# +
dataset_path = "../../Datasets/TII-SSRC-23 Dataset/pcap/"
pcapsToCSVs(
    dataset_path,
    [
        "benign/audio/audio.pcap",
        "benign/background/background.pcap",
        "benign/text/text.pcap",
        "benign/video/http.pcap",
        "benign/video/rtp.pcap",
        "benign/video/udp.pcap",
    ],
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Benign",
    5000000,
    {"Benign": {}},
    0,
)

pcapsToCSVs(
    dataset_path,
    [
        "malicious/bruteforce/bruteforce_dns.pcap",
        "malicious/bruteforce/bruteforce_ftp.pcap",
        "malicious/bruteforce/bruteforce_http.pcap",
        "malicious/bruteforce/bruteforce_ssh.pcap",
        "malicious/bruteforce/bruteforce_telnet.pcap",
    ],
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/Bruteforce",
    5000000,
    {"Bruteforce": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    [
        "malicious/dos/ack_tcp_dos.pcap",
        "malicious/dos/cwr_tcp_dos.pcap",
        "malicious/dos/ecn_tcp_dos.pcap",
        "malicious/dos/http_dos.pcap",
        "malicious/dos/icmp_dos.pcap",
        "malicious/dos/mac_dos.pcap",
        "malicious/dos/psh_tcp_dos.pcap",
        "malicious/dos/rst_tcp_dos.pcap",
        "malicious/dos/syn_tcp_dos.pcap",
        "malicious/dos/udp_dos.pcap",
        "malicious/dos/urg_tcp_dos.pcap",
        "malicious/mirai-botnet/mirai_ddos_ack.pcap",
        "malicious/mirai-botnet/mirai_ddos_dns.pcap",
        "malicious/mirai-botnet/mirai_ddos_greeth.pcap",
        "malicious/mirai-botnet/mirai_ddos_greip.pcap",
        "malicious/mirai-botnet/mirai_ddos_http.pcap",
        "malicious/mirai-botnet/mirai_ddos_syn.pcap",
        "malicious/mirai-botnet/mirai_ddos_udp_udpplain.pcap",
    ],  # "Attack traffic/DDoS ICMP Flood Attacks.pcap" is not included because I only look at UDP and TCP packets
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/DOS",
    5000000,
    {"DOS": {}},
    0,
)
pcapsToCSVs(
    dataset_path,
    ["malicious/information-gathering/information_gathering.pcap"],
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/Information Gathering",
    5000000,
    {"Information Gathering": {}},
    0,
)
# -

num_nodes = 20
csvs_paths = [
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Benign/",
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/Bruteforce",
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/DOS",
    "../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/CSVs/Malicious/Information Gathering",
]
graphs_path = (
    f"../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/Graphs/Size {num_nodes}/"
)
loadCSVsAndCreateGraphs(csvs_paths, graphs_path, num_nodes, 10000, "Generalized", "all")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_nodes = 20
epochs = 10
batch_size = 512
out_feats = 32
num_heads = 4

# +
one_hot_mapping = {
    "Benign": 1,
    "Bruteforce": 2,
    "DOS": 3,
    "Information Gathering": 4,
}
graphs_path = (
    f"../../Datasets/TII-SSRC-23 Dataset/My Preprocessing/Graphs/Size {number_nodes}/"
)
graphs, labels = loadGraphDataset(graphs_path, one_hot_mapping, 10)
(
    graphs_train,
    graphs_dev,
    graphs_test,
    labels_train,
    labels_dev,
    labels_test,
) = splitGraphDataset(graphs, labels, 0.1, 0.1, True, device)
# displayGraph(graphs_train[0])

attack_classification_best_model_params_path = os.path.join(
    "../../Checkpoints/",
    f"gat-tii-ssrc-23-multiclass-{number_nodes}-{num_heads}-{out_feats}.pt",
)

# +
# Train attack classification model
print("Training attack classification model")

evaluation_mode = {
    "mode": "train-test-dev",
    "set": "train",
    "name": "gat-tii-ssrc-23-multiclass",
}

attack_classification_model = GAT(
    number_nodes=number_nodes,
    number_features=4,
    out_feats=out_feats,
    batch_size=batch_size,
    dropout=0.2,
    num_classes=4,
    num_heads=num_heads,
    device=device,
).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(attack_classification_model.parameters(), lr=0.001)
best_train_loss = float("inf")
train_loss = float("inf")

for epoch_num in range(1, epochs + 1):
    graphs_train, labels_train = randomizeGraphOrder(graphs_train, labels_train)
    epoch_start_time = time.time()
    # print(torch.cat((labels_train[:, :1], torch.flip(labels_train[:, :1], [1])), dim=1))
    train_loss = train(
        attack_classification_model,
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
            attack_classification_model.state_dict(),
            attack_classification_best_model_params_path,
        )
print("=" * 89)
print("| Saving final checkpoint")
print("=" * 89)
torch.save(
    attack_classification_model.state_dict(),
    attack_classification_best_model_params_path,
)

evaluation_mode["set"] = "dev"
print("Evaluating attack classification model on dev set")
results = evaluate(
    attack_classification_model,
    attack_classification_best_model_params_path,
    loss_function,
    graphs_dev,
    labels_dev,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
with open(
    "Results/Pickle/gat-tii-ssrc-23-multiclass-results-dev.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

evaluation_mode["set"] = "test"
print("Evaluating attack classification model on test set")
results = evaluate(
    attack_classification_model,
    attack_classification_best_model_params_path,
    loss_function,
    graphs_test,
    labels_test,
    list(one_hot_mapping.keys()),
    batch_size,
    device,
    evaluation_mode,
)
print(f"The size of the model is {get_model_size_mb(attack_classification_model)}MB")
with open(
    "Results/Pickle/gat-tii-ssrc-23-multiclass-results-test.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

# Show dev set results
print("=" * 89)
print("Dev set metrics")
print("=" * 89)
with open("Results/Pickle/gat-tii-ssrc-23-multiclass-results-dev.pkl", "rb") as file:
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
with open("Results/Pickle/gat-tii-ssrc-23-multiclass-results-test.pkl", "rb") as file:
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
