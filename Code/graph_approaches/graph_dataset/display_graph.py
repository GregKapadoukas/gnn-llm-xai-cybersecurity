import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def displayGraph(G):
    # print(f"Nodes of graph: {G.nodes()}")
    # print(f"Edges of graph: {G.edges()}")
    # print(f"Node features: {G.ndata['feature']}")
    options = {
        "node_color": "black",
        "node_size": 20,
        "width": 1,
    }
    g = dgl.to_networkx(G.cpu())
    plt.figure(figsize=[15, 7])
    nx.draw(g, **options)
    plt.show()
    G.to(device)
