import math
import os
import pickle

import dgl
import numpy as np
import torch
from natsort import natsorted
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def convertToDynamicGraphs(dgl_graphs_path, temporal_graphs_path, num_time_steps):
    dgl_graphs_files = natsorted(os.listdir(dgl_graphs_path))
    dgl_file_counter = 0
    attacks = set(getAttackFromFilename(file) for file in dgl_graphs_files)
    attacks_to_file_coutner_dictionary = {attack: 0 for attack in attacks}
    for file in dgl_graphs_files:
        print(
            f"Converting Normal Graphs to Dynamic Graphs from file {file}: ({dgl_file_counter + 1}/{len(dgl_graphs_files)})"
        )
        dgl_file_counter += 1
        dgl_graphs, _ = dgl.load_graphs(dgl_graphs_path + file)
        num_batches = math.ceil(len(dgl_graphs) / num_time_steps) - 1
        assert len(dgl_graphs) >= num_time_steps
        dynamic_graphs = []
        features = []
        edges = []
        time_step_count = 0
        for i in range(num_batches):
            dgl_graphs_batch = getGraphsBatch(dgl_graphs, i, num_time_steps)
            for dgl_graph in dgl_graphs_batch:
                features.append(dgl_graph.ndata["feature"].to(torch.float).numpy())
                edges.append(np.array(dgl_graph.edges()))
                time_step_count += 1
                if time_step_count == num_time_steps:
                    dynamic_graphs.append(
                        DynamicGraphTemporalSignal(
                            edge_indices=edges,
                            edge_weights=[
                                np.ones(edge.shape, dtype=np.int32) for edge in edges
                            ],  # Not used
                            features=features,
                            targets=[
                                np.ones(f.shape, dtype=np.int32) for f in features
                            ],  # Not used
                        )
                    )
                    time_step_count = 0
                    features = []
                    edges = []
        file_count = attacks_to_file_coutner_dictionary[getAttackFromFilename(file)]
        with open(
            temporal_graphs_path
            + str(file_count)
            + "_"
            + getAttackFromFilename(file)
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(dynamic_graphs, f)
        attacks_to_file_coutner_dictionary[getAttackFromFilename(file)] += 1


def getGraphsBatch(graphs, i: int, num_time_steps):
    seq_len = min(num_time_steps, len(graphs) - 1 - i * num_time_steps)
    graphs_batch = graphs[i * num_time_steps : i * num_time_steps + seq_len]
    return graphs_batch


def getAttackFromFilename(file):
    index = file.find("_")
    return file[index + 1 : -4]
