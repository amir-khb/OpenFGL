"""
Analyze how layer_idx affects algorithm parameters for FedALA and NewALA.

This script shows what parameters are included for different layer_idx values.
"""

import torch
import numpy as np
import openfgl.config as config
from openfgl.utils.task_utils import load_node_edge_level_default_model
from openfgl.data.distributed_dataset_loader import FGLDataset


def estimate_ala_parameters(model, layer_idx):
    """Estimate FedALA parameters and show breakdown."""
    params = list(model.parameters())
    params_p = params[-layer_idx:]

    print(f"\n  Parameters included (last {layer_idx} tensors):")
    total = 0
    for i, param in enumerate(params_p):
        param_count = param.numel()
        total += param_count
        shape_str = str(tuple(param.shape))
        print(f"    [{len(params)-layer_idx+i}] Shape: {shape_str:<20} Params: {param_count:>8,}")

    print(f"  FedALA algo params: {total:,}")
    return total


def estimate_newala_parameters(model, layer_idx, rank):
    """Estimate NewALA parameters and show breakdown."""
    params = list(model.parameters())
    params_p = params[-layer_idx:]

    print(f"\n  Parameters included (last {layer_idx} tensors):")
    total = 0
    for i, param in enumerate(params_p):
        shape_str = str(tuple(param.shape))
        if param.dim() >= 2:
            if param.dim() == 2:
                m, n = param.shape
            else:
                m = param.shape[0]
                n = np.prod(param.shape[1:])

            r = min(rank, min(m, n))
            lora_params = n * r + m * r
            total += lora_params
            print(f"    [{len(params)-layer_idx+i}] Shape: {shape_str:<20} " +
                  f"LoRA: {n}×{r} + {m}×{r} = {lora_params:>8,}")
        else:
            print(f"    [{len(params)-layer_idx+i}] Shape: {shape_str:<20} " +
                  f"(1D - uses uniform weights, 0 params)")

    print(f"  NewALA algo params: {total:,}")
    return total


def main():
    args = config.args

    # Configuration
    args.root = "./dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 5
    args.task = "node_cls"
    args.dataset = ["Cora"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5
    args.newala_rank = 4

    print("=" * 100)
    print("Layer Index Analysis for FedALA and NewALA")
    print("=" * 100)
    print(f"Model: GCN with {args.num_layers} layers, hidden_dim={args.hid_dim}")
    print()

    # Load dataset and model
    print("Loading dataset...")
    fgl_dataset = FGLDataset(args)
    sample_data = fgl_dataset.local_data[0]

    num_features = sample_data.x.shape[1]
    num_classes = fgl_dataset.global_data.num_global_classes

    print(f"Input features: {num_features}, Output classes: {num_classes}\n")

    # Load model
    args.fl_algorithm = "fedavg"
    model = load_node_edge_level_default_model(args, input_dim=num_features, output_dim=num_classes)

    # Show all parameters
    print("=" * 100)
    print("BASE MODEL PARAMETERS")
    print("=" * 100)
    params = list(model.parameters())
    total_params = 0

    for i, param in enumerate(params):
        param_count = param.numel()
        total_params += param_count
        param_type = "Weight" if param.dim() >= 2 else "Bias"
        shape_str = str(tuple(param.shape))
        print(f"  [{i}] {param_type:<8} Shape: {shape_str:<20} Params: {param_count:>8,}")

    print(f"\n  TOTAL BASE PARAMS: {total_params:,}")

    # Analyze different layer_idx values
    layer_indices = [1, 2, 3, 4]

    print("\n" + "=" * 100)
    print("ALGORITHM PARAMETERS FOR DIFFERENT LAYER_IDX VALUES")
    print("=" * 100)

    results = []

    for layer_idx in layer_indices:
        print(f"\n{'='*100}")
        print(f"layer_idx = {layer_idx}")
        print(f"{'='*100}")

        fedala_params = estimate_ala_parameters(model, layer_idx)
        newala_params = estimate_newala_parameters(model, layer_idx, args.newala_rank)

        if fedala_params > 0:
            reduction = ((fedala_params - newala_params) / fedala_params * 100)
            ratio = fedala_params / newala_params if newala_params > 0 else float('inf')
        else:
            reduction = 0
            ratio = 0

        results.append({
            "layer_idx": layer_idx,
            "fedala": fedala_params,
            "newala": newala_params,
            "reduction": reduction,
            "ratio": ratio
        })

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()

    # Print table header
    print(f"{'layer_idx':<12} {'FedALA Params':<18} {'NewALA Params':<18} {'Reduction':<15} {'Ratio':<10}")
    print("-" * 100)

    for r in results:
        print(f"{r['layer_idx']:<12} {r['fedala']:>15,}   {r['newala']:>15,}   "
              f"{r['reduction']:>12.2f}%   {r['ratio']:>7.2f}x")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print("""
1. layer_idx=1 or 2: Only applies to output layer
   - Fastest, lowest memory
   - May be sufficient for small heterogeneity

2. layer_idx=3 or 4: Applies to multiple/all layers
   - More expressive adaptive aggregation
   - Better for high heterogeneity
   - Higher computational cost

3. For NewALA, even with layer_idx=4 (all params):
   - Still uses 5-10x fewer params than FedALA
   - Due to low-rank decomposition (rank=4)

Current setting (layer_idx=2):
   - Only adapting the output layer (455 params)
   - Very efficient but limited expressiveness
   - Consider layer_idx=4 for full model adaptation
    """)
    print("=" * 100)


if __name__ == "__main__":
    main()
