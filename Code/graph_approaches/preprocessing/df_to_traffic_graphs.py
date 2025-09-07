import hashlib

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch
from joblib import Parallel, delayed


def createGraphLayer(node_ids, src_nodes, dst_nodes):
    i = 0
    prev_node_id = node_ids[0]
    for node_id in node_ids:
        if i == 0:
            i += 1
            continue
        src_nodes.append(prev_node_id)
        dst_nodes.append(node_id)
        prev_node_id = node_id
        i += 1


def setPairFirstLayer(layer, pair_first_layer):
    if pair_first_layer == []:
        pair_first_layer = layer
    return pair_first_layer


def connectGraphLayers(layer1, layer2, src_nodes, dst_nodes, pair_first_layer):
    if (len(layer2)) > 0:
        src_nodes.append(layer1[0])
        dst_nodes.append(layer2[0])
        src_nodes.append(layer1[len(layer1) - 1])
        dst_nodes.append(layer2[len(layer2) - 1])
    pair_first_layer = setPairFirstLayer(layer2, pair_first_layer)
    return pair_first_layer


def mapProtocolFlags(flags):
    # Flags are one-hot encoded converted from final binary number to decimal
    if flags[0:3] == "UDP":
        result = 0
    else:
        result = 1 * 2 ^ 0
        if "F" in flags[4:]:
            result += 2 * 2 ^ 9
        if "S" in flags[4:]:
            result += 2 * 2 ^ 8
        if "R" in flags[4:]:
            result += 2 * 2 ^ 7
        if "P" in flags[4:]:
            result += 2 * 2 ^ 6
        if "A" in flags[4:]:
            result += 2 * 2 ^ 5
        if "U" in flags[4:]:
            result += 2 * 2 ^ 4
        if "E" in flags[4:]:
            result += 2 * 2 ^ 3
        if "C" in flags[4:]:
            result += 2 * 2 ^ 2
        if "N" in flags[4:]:
            result += 2 * 2 ^ 1
    return result


def createEndpointSubgraphs(
    df,
    seen_ip_pairs,
    pair_first_layer,
    previous_source_ip,
    currentGraphLayer,
    previousGraphLayer,
    src_nodes,
    dst_nodes,
    node_features,
):
    # Iterate for every packet
    for index, row in df.iterrows():
        # Compute the ip pair hash to separate the subtrees
        ip_pair = hashlib.sha256(
            str(sorted((row["Source IP"], row["Destination IP"]))).encode()
        ).hexdigest()
        # Initialize subtree if the ip pair hasn't been see before
        if ip_pair not in seen_ip_pairs:
            seen_ip_pairs.add(ip_pair)
            pair_first_layer[ip_pair] = []
            previous_source_ip[ip_pair] = row["Source IP"]
            currentGraphLayer[ip_pair] = [index]
            previousGraphLayer[ip_pair] = []
        # If the previous_source_ip is the same, the new packet should be appended to the current layer
        elif row["Source IP"] == previous_source_ip[ip_pair]:
            currentGraphLayer[ip_pair].append(index)
        # If the previous source IP is different, it means it is time to connect the current and previous layer and start a new current layer
        else:
            if len(previousGraphLayer[ip_pair]) != 0:
                pair_first_layer[ip_pair] = connectGraphLayers(
                    currentGraphLayer[ip_pair],
                    previousGraphLayer[ip_pair],
                    src_nodes,
                    dst_nodes,
                    pair_first_layer[ip_pair],
                )
            createGraphLayer(currentGraphLayer[ip_pair], src_nodes, dst_nodes)
            previousGraphLayer[ip_pair] = currentGraphLayer[ip_pair]
            currentGraphLayer[ip_pair] = [index]
        # Add the node features for the current node
        previous_source_ip[ip_pair] = row["Source IP"]
        new_node_features = torch.tensor(
            [
                row["Source Port"],
                row["Destination Port"],
                row["Length"],
                mapProtocolFlags(row["Protocol/Flags"]),
            ]
        ).view(1, -1)
        node_features.append(new_node_features)
    # The last layers have not yet been connected, so connect them
    for ip_pair in seen_ip_pairs:
        createGraphLayer(currentGraphLayer[ip_pair], src_nodes, dst_nodes)
        if len(previousGraphLayer[ip_pair]) != 0:
            pair_first_layer[ip_pair] = connectGraphLayers(
                currentGraphLayer[ip_pair],
                previousGraphLayer[ip_pair],
                src_nodes,
                dst_nodes,
                pair_first_layer[ip_pair],
            )
        # If no previous first layer has been set, set the current as first so it is connected in the final potentially combined graph
        pair_first_layer[ip_pair] = setPairFirstLayer(
            currentGraphLayer[ip_pair], pair_first_layer[ip_pair]
        )


def connectSubgraphs(
    seen_ip_pairs,
    pair_first_layer,
    src_nodes,
    dst_nodes,
):
    previous_ip_pair = None
    i = 1
    for ip_pair in seen_ip_pairs:
        if previous_ip_pair is None:
            previous_ip_pair = ip_pair
            first_ip_pair = ip_pair
        else:
            # Connect subgraphs with each other on the first layer
            connectGraphLayers(
                pair_first_layer[ip_pair],
                pair_first_layer[previous_ip_pair],
                src_nodes,
                dst_nodes,
                pair_first_layer[ip_pair],
            )
            previous_ip_pair = ip_pair
            # Connect last layer with first
            if i == len(seen_ip_pairs):
                connectGraphLayers(
                    pair_first_layer[ip_pair],
                    pair_first_layer[first_ip_pair],  # type: ignore
                    src_nodes,
                    dst_nodes,
                    pair_first_layer[ip_pair],
                )
        i += 1


def createTrafficGraph(df):
    df = df.reset_index()
    previous_source_ip = {}
    src_nodes = []
    dst_nodes = []
    currentGraphLayer = {}
    previousGraphLayer = {}
    node_features = []
    seen_ip_pairs = set()
    pair_first_layer = {}

    createEndpointSubgraphs(
        df,
        seen_ip_pairs,
        pair_first_layer,
        previous_source_ip,
        currentGraphLayer,
        previousGraphLayer,
        src_nodes,
        dst_nodes,
        node_features,
    )
    if len(seen_ip_pairs) > 1:
        connectSubgraphs(
            seen_ip_pairs,
            pair_first_layer,
            src_nodes,
            dst_nodes,
        )
    node_features = torch.cat(node_features, dim=0)
    G = dgl.graph((src_nodes, dst_nodes))
    G = dgl.to_bidirected(G)
    G.ndata["feature"] = node_features  # type: ignore
    return G


def dfToTrafficGraphs(df, n):
    num_graphs = df.shape[0] // n

    final_graphs = Parallel(n_jobs=-1)(
        delayed(createTrafficGraph)(
            df.iloc[i * n : (i + 1) * n],
        )
        for i in range(num_graphs)
    )
    return final_graphs
