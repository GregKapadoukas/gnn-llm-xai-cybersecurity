# Steps to download and install the IoT-23 dataset

1. Download the iot_23_datasets_full.tar.gz file form https://www.stratosphereips.org/datasets-iot23
2. Extract the file in this directory.
3. Once you have created the CSVs using the notebook, manually move the Benign CSVs from "CSVs/Benign" to "CSVs_sample/Benign" and a few Malicious CSVs from "CSVs/Malicious" to "CSVs_sample/Malicious" in order to combat the huge amount of malicious data. In the experiments conducted for the 'An Optimized Graph Transformer Network Attack Classifier' paper, all malicious CSV files were sampled with an increment of 15 (e.g., 0_packets.csv, 15_packets.csv, 30_packets.csv, ...)."
