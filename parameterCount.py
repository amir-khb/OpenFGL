import openfgl.config as config
from openfgl.utils.task_utils import load_node_edge_level_default_model
from openfgl.data.distributed_dataset_loader import FGLDataset
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def count_newala_params(model, layer_idx, rank):
    """
    Count NewALA parameters (Low-Rank Decomposition).
    Param count = (input_dim + output_dim) * rank
    """
    params = list(model.parameters())
    params_to_adapt = params[-layer_idx:]

    total = 0
    for param in params_to_adapt:
        if param.dim() >= 2:
            if param.dim() == 2:
                m, n = param.shape
            else:
                m = param.shape[0]
                n = np.prod(param.shape[1:])

            # NewALA decomposition size
            r = min(rank, min(m, n))
            total += (m + n) * r
        else:
            # Biases are usually kept as is or small enough to ignore difference,
            # but we count them for consistency
            total += param.numel()
    return total

def count_fedala_params(model, layer_idx):
    """
    Count FedALA parameters (Full Rank / Standard).
    Param count = input_dim * output_dim
    """
    params = list(model.parameters())
    params_to_adapt = params[-layer_idx:]

    total = 0
    for param in params_to_adapt:
        # FedALA keeps the full weight matrix for adaptation
        total += param.numel()
    return total

def analyze_client_scalability():
    """Analyze parameter scaling vs number of clients."""
    args = config.args

    # --- Configuration ---
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.task = "node_cls"
    args.dataset = ["PubMed"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # Analysis Settings
    layer_idx = 4  # Adapt all layers (2 weights + 2 biases)
    newala_rank = 32  # Fixed rank for comparison
    client_counts = [5, 10, 15, 20, 25, 30]

    print("="*80)
    print("SCALABILITY ANALYSIS: FedALA vs NewALA")
    print("="*80)
    print(f"Dataset: {args.dataset[0]}")
    print(f"Model: 2-Layer GCN (Hidden: {args.hid_dim})")
    print(f"NewALA Rank: {newala_rank}")
    print("="*80)

    # 1. Load Data & Model (Dummy run to get shapes)
    args.num_clients = 10 # Temporary just for loading
    fgl_dataset = FGLDataset(args)
    sample_data = fgl_dataset.local_data[0]
    num_features = sample_data.x.shape[1]
    num_classes = fgl_dataset.global_data.num_global_classes

    model = load_node_edge_level_default_model(args, input_dim=num_features, output_dim=num_classes)

    # 2. Calculate Base Costs (Per Client)
    fedala_per_client = count_fedala_params(model, layer_idx)
    newala_per_client = count_newala_params(model, layer_idx, newala_rank)

    reduction_pct = (1 - newala_per_client / fedala_per_client) * 100

    print(f"\n[Per-Client Memory Unit]")
    print(f"FedALA (Full): {fedala_per_client:,} params")
    print(f"NewALA (Rank={newala_rank}): {newala_per_client:,} params")
    print(f"Reduction: {reduction_pct:.2f}% per client")

    # 3. Analyze Scaling
    print("\n" + "="*80)
    print(f"TOTAL SYSTEM PARAMETERS (Sum of all {args.dataset[0]} clients)")
    print("="*80)

    table = PrettyTable()
    table.field_names = ["# Clients", "FedALA Total", "NewALA Total", "Saved Params", "Memory Saved (MB)"]
    table.align = "r"

    results = []

    for n_clients in client_counts:
        # Total parameters stored in the system (sum of all local ALA modules)
        total_fedala = fedala_per_client * n_clients
        total_newala = newala_per_client * n_clients

        saved = total_fedala - total_newala
        saved_mb = (saved * 4) / (1024 * 1024) # 4 bytes per float32

        table.add_row([
            n_clients,
            f"{total_fedala:,}",
            f"{total_newala:,}",
            f"{saved:,}",
            f"{saved_mb:.2f} MB"
        ])

        results.append({
            'clients': n_clients,
            'fedala': total_fedala,
            'newala': total_newala
        })

    print(table)

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"As the system scales to {client_counts[-1]} clients:")
    print(f"1. FedALA requires storing ~{results[-1]['fedala']/1e6:.1f}M parameters total.")
    print(f"2. NewALA requires only ~{results[-1]['newala']/1e6:.1f}M parameters total.")
    print(f"3. Using NewALA saves {results[-1]['fedala']/results[-1]['newala']:.1f}x storage space.")
    print("="*80)

if __name__ == "__main__":
    analyze_client_scalability()