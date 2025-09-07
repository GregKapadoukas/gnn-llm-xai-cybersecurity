import hashlib
import math
import os

import dgl
import pandas as pd
from natsort import natsorted

from preprocessing.df_to_traffic_graphs import dfToTrafficGraphs


def loadCSVsAndCreateGraphs(
    csvs_paths,
    graphs_path,
    packet_num_in_graph,
    graphs_threshold,
    graph_type,
    network_ips,
):
    assert graph_type == "Endpoint" or graph_type == "Generalized"
    graphs_by_label = {}
    unique_graph_hashes = set()
    for csvs_path in csvs_paths:
        print(f"Converting from path {csvs_path}")
        file_list = os.listdir(csvs_path)
        i = 0
        for file_name in natsorted(file_list):
            print(f"{i+1}/{len(file_list)} Processing file {file_name}")
            if file_name.endswith(".csv"):
                file_path = os.path.join(csvs_path, file_name)
                df = pd.read_csv(file_path)
                if not df.empty:
                    if graph_type == "Endpoint":
                        df_splitted = splitByIPPairAndLabel(df, network_ips)
                    else:
                        df_splitted = splitByIPAndLabel(df, network_ips)
                    del df
                    for key in df_splitted:
                        label = df_splitted[key]["Label"][0]
                        # print(
                        #    f"Processing label: {label}, with graphs len size: {math.floor(len(df_splitted[key])/packet_num_in_graph)}"
                        # )
                        graphs = dfToTrafficGraphs(
                            df_splitted[key], packet_num_in_graph
                        )
                        graphs, unique_graph_hashes = removeIdenticalGraphsFromList(
                            graphs, unique_graph_hashes
                        )
                        graphs_by_label = groupGraphsByLabel(
                            graphs_by_label,
                            graphs_path,
                            graphs_threshold,
                            graphs,
                            label,
                        )
                    i += 1
    graphs_by_label = processRemainingGraphs(graphs_by_label, graphs_path)


def groupGraphsByLabel(graphs_by_label, graphs_path, graphs_threshold, graphs, label):
    if label not in graphs_by_label:
        graphs_by_label[label] = [0, graphs]
    else:
        graphs_by_label[label][1] += graphs
    while len(graphs_by_label[label][1]) > graphs_threshold:
        # Create same sized graphs
        dgl.save_graphs(
            graphs_path + f"{graphs_by_label[label][0]}_{label}.dgl",
            graphs_by_label[label][1][:graphs_threshold],
        )
        graphs_by_label[label][0] += 1
        graphs_by_label[label][1] = graphs_by_label[label][1][graphs_threshold:]
    return graphs_by_label


def processRemainingGraphs(graphs_by_label, graphs_path):
    if graphs_by_label:
        for label in graphs_by_label:
            if graphs_by_label[label][1]:
                if graphs_by_label[label][1] is not []:
                    dgl.save_graphs(
                        graphs_path + f"{graphs_by_label[label][0]}_{label}.dgl",
                        graphs_by_label[label][1],
                    )


def splitByIPPairAndLabel(df, network_ips):
    if network_ips != "all":
        df = df[
            (df["Source IP"].isin(network_ips))
            | (df["Destination IP"].isin(network_ips))
        ]
    df["Split ID"] = df.apply(
        lambda row: sorted([row["Source IP"], row["Destination IP"], row["Label"]]),
        axis=1,
    )
    df = df.astype({"Split ID": "string"})
    df_by_label_grouped = df.groupby("Split ID")
    df_by_label_sub_dfs = {}
    for name, group in df_by_label_grouped:
        df_by_label_sub_dfs[name] = group
        df_by_label_sub_dfs[name].pop("Split ID")
        df_by_label_sub_dfs[name].reset_index(inplace=True)
    return df_by_label_sub_dfs


def splitByIPAndLabel(df, network_ips):
    df_by_ip_and_label = {}
    unique_ips = pd.concat([df["Source IP"], df["Destination IP"]]).unique()
    if network_ips != "all":
        unique_ips = list(set(unique_ips).intersection(network_ips))
    for ip in unique_ips:
        ip_df = df[(df["Source IP"] == ip) | (df["Destination IP"] == ip)]
        ip_df_by_label = ip_df.groupby("Label")
        for label, value in ip_df_by_label:
            df_by_ip_and_label[ip + "," + label] = value
    for key in df_by_ip_and_label.keys():
        df_by_ip_and_label[key].reset_index(inplace=True)
    return df_by_ip_and_label


def hashGraphContent(g):
    node_features_str = str(g.ndata["feature"].tolist())
    edge_indices_str = str(g.edges()[0].tolist()) + str(g.edges()[1].tolist())
    combined_str = node_features_str + edge_indices_str
    return hashlib.sha256(combined_str.encode()).hexdigest()


def removeIdenticalGraphsFromList(graphs, unique_graph_hashes):
    unique_graphs = []
    for graph in graphs:
        hash = hashGraphContent(graph)
        if hash not in unique_graph_hashes:
            unique_graph_hashes.add(hash)
            unique_graphs.append(graph)
    return unique_graphs, unique_graph_hashes
