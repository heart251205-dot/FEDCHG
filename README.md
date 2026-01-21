FedCHG: Graph Autoencoder Enhanced Federated Learning for Cross-Domain Heterogeneous Graph
This repository contains the official PyTorch implementation of FedCHG, a novel Federated Graph Learning (FGL) framework designed to tackle Cross-Domain Heterogeneity

FedCHG_Repo/
├── run_exp.py              # [Key Script] Automated scheduler 
├── requirements.txt        # Python environment dependencies
├── README.md               # You are here
├── scripts/
│   └── download_data.sh    # Helper script for downloading large datasets (DGraph/Reddit)
└── src/
    ├── __init__.py
    ├── main.py             # Main entry point for single-scenario experiments  
    ├── client.py           # Client-side logic 
    ├── server.py           # Server-side logic 
    ├── models.py           # Model definitions 
    ├── features.py         # Structural Feature Extraction 
    ├── data_loader.py      # Dataset loading, splitting, and heterogeneity simulation
    ├── training.py         # Training
    └── utils.py            # Metrics, Logging, T-Test, and Visualization tools


🛠️ Environment Setup
Dependencies are sensitive for Graph Neural Networks. Please install strictly according to the versions below to avoid torch_geometric compatibility issues.
Create a Conda Environment:
Bash
conda create -n fedchg python=3.8.13
conda activate fedchg



Install Dependencies:
Bash
pip install -r requirements.txt

Key libraries: torch-geometric==2.0.4, numpy, scipy, scikit-learn, seaborn.



