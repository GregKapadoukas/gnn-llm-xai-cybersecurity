# %%
import dgl
import torch

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ldap_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_LDAP.dgl"
)
mssql_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_MSSQL.dgl"
)
netbios_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_NetBIOS.dgl"
)
ntp_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_NTP.dgl"
)
portmap_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_PortMap.dgl"
)
webddos_graphs, labels = dgl.load_graphs(
    "../../Datasets/CIC-DDOS2019/My Preprocessing/Graphs_test/Size 20/0_WebDDoS.dgl"
)
# %%
print(f"LDAP samples: {len(ldap_graphs)}")
print(f"MSSQL samples: {len(mssql_graphs)}")
print(f"NetBIOS samples: {len(netbios_graphs)}")
print(f"NTP samples: {len(ntp_graphs)}")
print(f"PortMap samples: {len(portmap_graphs)}")
print(f"WebDDoS samples: {len(webddos_graphs)}")

