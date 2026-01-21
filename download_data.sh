#!/bin/bash

# ==============================================================================
# FedCHG Data Download Helper Script
# ==============================================================================
# This script assists in downloading and setting up large-scale datasets 
# required for the FIN-SOC (Finance & Social) scenario.
#
# Datasets covered:
# 1. Reddit (Social Network) - Source: SNAP / TUDataset
# 2. DGraph (Finance) - Source: https://dgraph.xinye.com/
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
DATA_ROOT="./data"
FIN_SOC_DIR="${DATA_ROOT}/FIN-SOC"
REDDIT_DIR="${FIN_SOC_DIR}/Reddit"
DGRAPH_DIR="${FIN_SOC_DIR}/DGraph"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting FedCHG Data Setup ===${NC}"

# 1. Check Dependencies
echo -e "${YELLOW}[1/4] Checking system dependencies...${NC}"
for cmd in wget unzip tar; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: '$cmd' is not installed. Please install it and retry.${NC}"
        exit 1
    fi
done
echo "Dependencies OK."

# 2. Create Directory Structure
echo -e "${YELLOW}[2/4] Creating directory structure...${NC}"
mkdir -p "$REDDIT_DIR"
mkdir -p "$DGRAPH_DIR"
echo "Created: $FIN_SOC_DIR"

# 3. Download Reddit Dataset
# Note: For graph classification, TUDataset usually handles this automatically.
# However, for the specific FIN-SOC scenario described in the paper (Hamilton et al., 2017),
# we provide the raw binary download logic here as a fallback or for pre-processing.
echo -e "${YELLOW}[3/4] Processing Reddit Dataset...${NC}"

if [ -f "${REDDIT_DIR}/REDDIT-BINARY.zip" ] || [ -f "${REDDIT_DIR}/REDDIT-BINARY/graph_labels.txt" ]; then
    echo "Reddit dataset already exists. Skipping download."
else
    echo "Downloading REDDIT-BINARY from TUDataset repository..."
    # Using the standard TUDataset link which is reliable
    wget -c https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip -O "${REDDIT_DIR}/REDDIT-BINARY.zip"
    
    echo "Extracting Reddit..."
    unzip -q "${REDDIT_DIR}/REDDIT-BINARY.zip" -d "$REDDIT_DIR"
    
    # Cleanup
    # rm "${REDDIT_DIR}/REDDIT-BINARY.zip"
    echo -e "${GREEN}Reddit Setup Complete.${NC}"
fi

# 4. Handle DGraph Dataset (Manual Action Required)
# DGraph is a financial dataset that often requires user agreement or specific access.
echo -e "${YELLOW}[4/4] Processing DGraph Dataset...${NC}"

if [ -f "${DGRAPH_DIR}/dgraph_fin.npz" ] || [ -f "${DGRAPH_DIR}/processed/data.pt" ]; then
    echo "DGraph dataset detected."
else
    echo -e "${RED}![IMPORTANT] DGraph Dataset Action Required${NC}"
    echo "Due to license restrictions and privacy policies of financial data,"
    echo "we cannot provide a direct download link for DGraph."
    echo ""
    echo "Please follow these steps to download:"
    echo "  1. Visit the official website: https://dgraph.xinye.com/"
    echo "  2. Download the 'DGraph-Fin' dataset."
    echo "  3. Place the 'dgraph_fin.npz' file into: ${DGRAPH_DIR}/"
    echo ""
    echo "Creating a placeholder file to prevent runtime crash..."
    touch "${DGRAPH_DIR}/README_DOWNLOAD_INSTRUCTIONS.txt"
fi

# 5. Final Message
echo -e "${GREEN}=== Data Setup Finished ===${NC}"
echo "Summary:"
echo " - Reddit: Ready in $REDDIT_DIR"
echo " - DGraph: Please ensure manual download is placed in $DGRAPH_DIR"
echo ""
echo "You can now run the experiment using:"
echo "  python run_exp.py --scenario FIN-SOC"