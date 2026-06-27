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
    n_jobs: int = -1,
):
    assert graph_type == "Endpoint" or graph_type == "Generalized"
    graphs_by_label = {}
    traffic_graph_stats = {
        "packet_rows_seen": 0,
        "packet_rows_converted": 0,
        "packet_rows_dropped": 0,
        "graphs_created": 0,
        "elapsed_seconds": 0.0,
    }
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
                        graphs, df_to_graph_stats = dfToTrafficGraphs(
                            df_splitted[key], packet_num_in_graph, return_stats=True, n_jobs=n_jobs
                        )
                        for stat_name in traffic_graph_stats:
                            traffic_graph_stats[stat_name] += df_to_graph_stats[stat_name]
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
    printTrafficGraphStats(traffic_graph_stats)


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


def printTrafficGraphStats(stats):
    elapsed_seconds = stats["elapsed_seconds"]
    graphs_per_second = (
        stats["graphs_created"] / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    packets_per_second = (
        stats["packet_rows_converted"] / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )
    avg_graph_latency_ms = (
        (elapsed_seconds / stats["graphs_created"]) * 1000
        if stats["graphs_created"] > 0
        else 0.0
    )
    print("=" * 89)
    print(
        "| DataFrame-to-traffic-graph preprocessing\n"
        f"| packet rows seen: {stats['packet_rows_seen']} "
        f"| packet rows converted: {stats['packet_rows_converted']} "
        f"| packet rows dropped by incomplete graph windows: {stats['packet_rows_dropped']}\n"
        f"| graphs created before duplicate removal: {stats['graphs_created']} "
        f"| graph construction time: {elapsed_seconds:.6f} s\n"
        f"| avg graph construction latency: {avg_graph_latency_ms:.6f} ms/graph\n"
        f"| throughput: {graphs_per_second:.6f} graphs/s "
        f"| {packets_per_second:.6f} packet rows/s"
    )
    print("=" * 89)
