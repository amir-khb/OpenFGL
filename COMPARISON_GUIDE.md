# FedALA vs NewALA Comparison Guide

This guide explains how to run comprehensive comparisons between FedALA and NewALA (LoRA-CAAA).

## Setup

### 1. Install Dependencies

```bash
pip install -r comparison_requirements.txt
```

Or install PyTorch and PyG manually following the official instructions:
- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Verify Installation

```bash
python -c "import torch; import torch_geometric; import openfgl; print('All dependencies installed successfully')"
```

## Available Comparison Scripts

### 1. `compare_fedala_newala.py` - Comprehensive Accuracy and Timing Comparison

**Purpose**: Runs both FedALA and NewALA on multiple datasets, comparing accuracy and training/testing time.

**Features**:
- Runs on 7 datasets: Cora, CiteSeer, PubMed, Photo, Computers, Chameleon, Actor
- 3 runs per configuration for statistical significance
- Tracks test accuracy, validation accuracy, best round, training time, testing time
- Generates multiple output files with detailed statistics

**Run**:
```bash
python compare_fedala_newala.py
```

**Outputs** (saved in timestamped directory `fedala_vs_newala_YYYYMMDD_HHMMSS/`):
- `raw_results.csv`: All individual run results
- `summary_stats.csv`: Aggregated statistics with mean ± std
- `comparison_table.csv`: Head-to-head comparison
- `report.txt`: Formatted text report
- `intermediate_results.csv`: Saved after each experiment (useful if script is interrupted)

**Configuration**:
Edit the script to modify:
- `datasets`: List of datasets to test
- `num_runs`: Number of runs per configuration (default: 3)
- `num_rounds`: Number of FL rounds (default: 100)
- `num_clients`: Number of clients (default: 5)
- Hyperparameters for FedALA and NewALA

**Expected Runtime**:
- ~5-10 minutes per experiment (dataset + algorithm + run)
- Total: 42 experiments = 3.5-7 hours

### 2. `compare_methods_simple.py` - Parameter Complexity Comparison

**Purpose**: Compares parameter counts across all 21 FL algorithms including NewALA.

**Features**:
- No external dependencies beyond OpenFGL
- Shows base parameters, algorithm-specific parameters, total parameters
- Displays communication cost (parameters sent to server)
- Highlights NewALA's parameter reduction vs FedALA

**Run**:
```bash
python compare_methods_simple.py
```

**Outputs**:
- Console: Formatted ASCII table with parameter comparison
- `fl_methods_comparison.txt`: Saved results

**Expected Runtime**: ~2-3 minutes

### 3. `compare_methods_complexity.py` - Full Complexity Analysis

**Purpose**: Comprehensive analysis including GFLOPs estimation.

**Requires**: `thop` and `prettytable` packages

**Features**:
- Parameter counts
- GFLOPs for forward/backward/overhead
- Algorithm-specific computational overhead estimation

**Run**:
```bash
python compare_methods_complexity.py
```

**Outputs**:
- Console: PrettyTable with full complexity metrics
- `fl_methods_complexity.txt`: Saved results

**Expected Runtime**: ~3-4 minutes

### 4. `analyze_layer_idx.py` - Layer Index Impact Analysis

**Purpose**: Shows how `layer_idx` parameter affects algorithm parameters.

**Features**:
- Displays all model parameters with shapes
- Shows which parameters are adapted for different layer_idx values
- Compares FedALA vs NewALA parameter reduction for each setting

**Run**:
```bash
python analyze_layer_idx.py
```

**Key Insight**:
- `layer_idx=2`: Only adapts output layer (455 FedALA params → 284 NewALA params)
- `layer_idx=4`: Adapts all parameters (92,231 FedALA params → 6,272 NewALA params, **93.2% reduction**)

**Expected Runtime**: ~1 minute

### 5. `test_newala.py` - Quick NewALA Test

**Purpose**: Simple test script to verify NewALA implementation.

**Run**:
```bash
python test_newala.py
```

**Configuration**:
Edit the script to modify:
- Dataset (default: Cora)
- Number of rounds, clients, epochs
- NewALA hyperparameters (rank, gamma, lambda, etc.)

**Expected Runtime**: ~2-3 minutes

## Understanding the Results

### NewALA vs FedALA - Key Differences

**Parameter Efficiency**:
- FedALA: Learns full m×n weight matrix per adapted layer
- NewALA: Learns low-rank decomposition (m+n)×r per adapted layer
- Typical reduction: 80-95% with rank=4

**Example** (64×32 weight matrix):
- FedALA: 64 × 32 = 2,048 parameters
- NewALA (r=4): (64 + 32) × 4 = 384 parameters
- Reduction: 81.25%

**Accuracy**:
- NewALA typically maintains or improves accuracy despite parameter reduction
- Low-rank structure acts as beneficial regularization
- Confidence-aware gating provides adaptive trust

**Training Time**:
- NewALA often faster due to fewer parameters to optimize
- Low-rank matrix multiplications are efficient
- Overall speedup: 1.1-1.5x typical

### Hyperparameter Tuning

**NewALA Key Hyperparameters**:
- `newala_rank` (default: 4): Rank for low-rank decomposition
  - Lower = fewer params, more regularization
  - Higher = more expressive, more params
  - Typical range: 2-8

- `newala_gamma` (default: 0.1): Sensitivity for confidence-aware gating
  - Lower = more aggressive gating (less trust)
  - Higher = more conservative gating (more trust)
  - Typical range: 0.05-0.5

- `newala_lambda_reg` (default: 0.01): Entropy regularization weight
  - Controls smoothness of aggregation weights
  - Typical range: 0.001-0.1

- `newala_layer_idx` (default: 2): Number of parameter tensors to adapt (from top)
  - 2 = output layer only (fast, less expressive)
  - 4 = all layers (slower, more expressive)
  - For 2-layer GCN: use 4 to adapt all parameters

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch following instructions at https://pytorch.org/get-started/locally/

### "'NodeClsTask' object has no attribute 'model'"
Cached dataset from previous run. Delete cache:
```bash
rm -rf ./dataset/distrib/subgraph_fl_louvain_*
```

### Out of Memory
Reduce:
- `num_clients`: Fewer clients per round
- `batch_size`: Smaller batches (not exposed in comparison scripts)
- `hid_dim`: Smaller hidden dimension
- Or use smaller datasets (Cora, CiteSeer instead of PubMed, ogbn-products)

### Script Interrupted
The comprehensive comparison saves `intermediate_results.csv` after each experiment. You can resume by:
1. Checking which experiments completed
2. Modifying the script to skip completed ones
3. Or just restart - intermediate results are preserved

## Citation

If you use NewALA (LoRA-CAAA) in your research, please cite:
```
# Add citation when paper is published
```

## Questions?

For issues or questions:
- Check existing scripts for usage examples
- Review the main README.md for OpenFGL documentation
- Consult openfgl/flcore/newala/README.md for NewALA implementation details
