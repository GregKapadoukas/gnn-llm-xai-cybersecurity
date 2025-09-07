# "Efficient deployment of graph and LLM based techniques in anomaly and network attack detection" Experiments Code

## Description

This repository contains the source code for the experiments done for the paper "Efficient deployment of graph and LLM based techniques in anomaly and network attack detection".

## Execution Environment Setup

1. Go to directory Datasets where you should follow the instructions to download and properly place the files from the datasets you want.
2. The code for graph_approaches, flow_approaches, and llm_approaches use different libraries, so you might want to use separate conda environments. The libraries we used are mentioned in the paper. For more info on conda, look up anaconda or miniconda.
3. The notebooks we provided can be run as is, as jupyter notebooks, or converted to plain .py code using jupytext.
4. Select and execute any notebook / python script.

## Extra Information

1. The graph models create checkpoints into the ../Checkpoints directory, and results in Pickle files and Confusion Diagrams into the Results directory. The flow models show all results in the notebook.
2. The graph models in the first cells of the notebooks preprocess the PCAPs into CSVs and into graphs. These steps store the results in files, and can be omitted in following executions.
3. If you get "ERROR: ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'" when trying to run the temporal models, you need to go to file: ~/miniconda3/envs/torch/lib/python3.12/site-packages/torch_geometric_temporal/nn/attention/tsagcn.py and change the line "from torch_geometric.utils.to_dense_adj import to_dense_adj" to "from torch_geometric.utils import to_dense_adj". This will fix the issue.
