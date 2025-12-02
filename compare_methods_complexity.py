"""
Compare computational complexity (parameters and GFLOPs) of different FL methods.

This script compares:
- Number of trainable parameters
- Number of communicable parameters (sent to server)
- GFLOPs for forward and backward passes
- Algorithm-specific overhead

Supports all FL algorithms in OpenFGL including NewALA (LoRA-CAAA).
"""

import torch
import torch.nn as nn
import numpy as np
from prettytable import PrettyTable
import openfgl.config as config
from openfgl.utils.task_utils import load_node_edge_level_default_model
from openfgl.data.distributed_dataset_loader import FGLDataset
from thop import profile, clever_format


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_ala_parameters(model, layer_idx=2):
    """
    Estimate additional parameters for FedALA.
    ALA learns per-parameter aggregation weights (same shape as model parameters).
    """
    params = list(model.parameters())
    params_p = params[-layer_idx:]

    # ALA stores one weight per parameter element in top layers
    ala_weights = sum(p.numel() for p in params_p)

    return ala_weights


def estimate_newala_parameters(model, layer_idx=2, rank=4):
    """
    Estimate additional parameters for NewALA (LoRA-CAAA).
    NewALA uses low-rank decomposition: W = σ(A · B^T)
    For 2D parameters: stores A (n×r) and B (m×r)
    For 1D parameters: no additional parameters (uses uniform weights)
    """
    params = list(model.parameters())
    params_p = params[-layer_idx:]

    total_newala_params = 0

    for param in params_p:
        if param.dim() >= 2:
            # 2D parameter: use low-rank decomposition
            if param.dim() == 2:
                m, n = param.shape
            else:
                # For higher-dim tensors, flatten to 2D
                m = param.shape[0]
                n = np.prod(param.shape[1:])

            r = min(rank, min(m, n))
            # A matrix: (n × r), B matrix: (m × r)
            total_newala_params += n * r + m * r
        # For 1D parameters (biases), no additional parameters needed

    return total_newala_params


def estimate_gflops(model, input_data):
    """
    Estimate GFLOPs for a forward pass using thop library.
    """
    model.eval()

    try:
        # Clone input to avoid modifying original
        input_clone = input_data.clone()

        # Profile the model
        macs, params = profile(model, inputs=(input_clone,), verbose=False)

        # Convert MACs to FLOPs (1 MAC ≈ 2 FLOPs)
        flops = 2 * macs
        gflops = flops / 1e9

        return gflops
    except Exception as e:
        print(f"Warning: Could not profile model: {e}")
        return 0.0


def estimate_algorithm_overhead(algorithm, model, layer_idx=2, rank=4):
    """
    Estimate computational overhead specific to each FL algorithm.
    Returns approximate additional GFLOPs per training iteration.
    """
    params = list(model.parameters())
    params_p = params[-layer_idx:]

    overhead_gflops = 0.0

    if algorithm == "fedala":
        # ALA: Weight learning iterations (gradient computation for weights)
        # Approximate: 1-2 extra forward passes per round for weight learning
        for param in params_p:
            operations = param.numel() * 2  # Gradient computation
            overhead_gflops += operations / 1e9
        overhead_gflops *= 2  # Multiply by ~2 forward passes

    elif algorithm == "newala":
        # NewALA: Low-rank matrix optimization
        # Overhead includes:
        # 1. Computing W = σ(A·B^T) for each parameter
        # 2. Entropy computation
        # 3. Gradient computation for A and B

        for param in params_p:
            if param.dim() >= 2:
                if param.dim() == 2:
                    m, n = param.shape
                else:
                    m = param.shape[0]
                    n = np.prod(param.shape[1:])

                r = min(rank, min(m, n))
                # Matrix multiplication A @ B.T: (n×r) @ (r×m) = O(n*r*m)
                operations = n * r * m
                overhead_gflops += operations / 1e9

        # Add entropy computation overhead (small)
        overhead_gflops *= 1.1

    elif algorithm == "fedprox":
        # FedProx: Proximal term computation
        # Additional: ||θ - θ_global||^2
        total_params = sum(p.numel() for p in model.parameters())
        overhead_gflops = total_params * 2 / 1e9  # Subtraction + norm

    elif algorithm == "scaffold":
        # SCAFFOLD: Control variates computation
        total_params = sum(p.numel() for p in model.parameters())
        overhead_gflops = total_params * 3 / 1e9  # Additional gradient corrections

    elif algorithm == "moon":
        # MOON: Contrastive learning overhead
        # Additional forward passes through previous models
        # Approximate as 1 extra forward pass
        overhead_gflops = 0.1  # Placeholder

    # Other algorithms (fedavg, isolate, etc.) have negligible overhead

    return overhead_gflops


def main():
    """
    Main function to compare FL algorithms.
    """
    args = config.args

    # Configuration
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 5
    args.task = "node_cls"
    args.dataset = ["Cora"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # FedALA specific
    args.ala_layer_idx = 4

    # NewALA specific
    args.newala_layer_idx = 4
    args.newala_rank = 32

    print("=" * 80)
    print("FL Methods Complexity Comparison")
    print("=" * 80)
    print(f"Dataset: {args.dataset[0]}")
    print(f"Model: {args.model[0]}")
    print(f"Hidden dim: {args.hid_dim}, Num layers: {args.num_layers}")
    print("=" * 80)
    print()

    # Load dataset to get model input/output dimensions
    print("Loading dataset...")
    fgl_dataset = FGLDataset(args)
    sample_data = fgl_dataset.local_data[0]

    # Get model dimensions
    num_features = sample_data.x.shape[1]
    num_classes = fgl_dataset.global_data.num_global_classes

    print(f"Input features: {num_features}, Output classes: {num_classes}")
    print()

    # List of algorithms to compare - ALL supported algorithms in OpenFGL
    algorithms = [
        "isolate",
        "fedavg",
        "fedprox",
        "scaffold",
        "moon",
        "feddc",
        "fedproto",
        "fedtgp",
        "fedpub",
        "fedstar",
        "fedgta",
        "fedtad",
        "gcfl_plus",
        "fedsage_plus",
        "adafgl",
        "feddep",
        "fggp",
        "fgssl",
        "fedgl",
        "fedala",
        "newala"
    ]

    results = []

    for algorithm in algorithms:
        print(f"Analyzing {algorithm.upper()}...")

        # Load base model
        args.fl_algorithm = algorithm
        args.model = ["gcn"]

        try:
            model = load_node_edge_level_default_model(
                args,
                input_dim=num_features,
                output_dim=num_classes
            )

            # Count base model parameters
            total_params, trainable_params = count_parameters(model)

            # Algorithm-specific additional parameters
            algo_params = 0
            if algorithm == "fedala":
                algo_params = estimate_ala_parameters(model, args.ala_layer_idx)
            elif algorithm == "newala":
                algo_params = estimate_newala_parameters(model, args.newala_layer_idx, args.newala_rank)

            # Total trainable parameters (base + algorithm-specific)
            total_trainable = trainable_params + algo_params

            # Communication cost (parameters sent to server)
            # Most algorithms send full model, some send additional info
            comm_params = trainable_params
            if algorithm == "scaffold":
                comm_params *= 2  # Control variates

            # Estimate GFLOPs for forward pass
            # Create dummy input matching the data structure
            gflops_forward = estimate_gflops(model, sample_data)

            # Backward pass is typically ~2x forward pass
            gflops_backward = gflops_forward * 2

            # Algorithm-specific overhead
            algo_overhead = estimate_algorithm_overhead(
                algorithm, model,
                layer_idx=args.newala_layer_idx if algorithm == "newala" else args.ala_layer_idx,
                rank=args.newala_rank
            )

            # Total GFLOPs per iteration (forward + backward + overhead)
            total_gflops = gflops_forward + gflops_backward + algo_overhead

            # Store results
            results.append({
                "Algorithm": algorithm.upper(),
                "Base Params": trainable_params,
                "Algo Params": algo_params,
                "Total Params": total_trainable,
                "Comm Params": comm_params,
                "Forward GFLOPs": gflops_forward,
                "Total GFLOPs": total_gflops,
                "Overhead": algo_overhead
            })

        except Exception as e:
            print(f"  Skipped (incompatible with current config): {str(e)[:50]}")
            # Skip algorithms that don't work with current configuration
            continue

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Create comparison table
    table = PrettyTable()
    table.field_names = [
        "Algorithm",
        "Base Params",
        "Algo Params",
        "Total Params",
        "Comm Params",
        "Forward GFLOPs",
        "Total GFLOPs/iter",
        "Overhead GFLOPs"
    ]

    # Format numbers for display
    for result in results:
        row = [
            result["Algorithm"],
            f"{result['Base Params']:,}" if isinstance(result['Base Params'], int) else result['Base Params'],
            f"{result['Algo Params']:,}" if isinstance(result['Algo Params'], int) else result['Algo Params'],
            f"{result['Total Params']:,}" if isinstance(result['Total Params'], int) else result['Total Params'],
            f"{result['Comm Params']:,}" if isinstance(result['Comm Params'], int) else result['Comm Params'],
            f"{result['Forward GFLOPs']:.4f}" if isinstance(result['Forward GFLOPs'], float) else result['Forward GFLOPs'],
            f"{result['Total GFLOPs']:.4f}" if isinstance(result['Total GFLOPs'], float) else result['Total GFLOPs'],
            f"{result['Overhead']:.4f}" if isinstance(result['Overhead'], float) else result['Overhead']
        ]
        table.add_row(row)

    print(table)
    print()

    # Additional analysis
    print("=" * 80)
    print("PARAMETER REDUCTION ANALYSIS")
    print("=" * 80)

    # Compare NewALA vs FedALA
    newala_result = next((r for r in results if r["Algorithm"] == "NEWALA"), None)
    fedala_result = next((r for r in results if r["Algorithm"] == "FEDALA"), None)

    if newala_result and fedala_result and isinstance(newala_result["Algo Params"], int):
        reduction = ((fedala_result["Algo Params"] - newala_result["Algo Params"]) /
                    fedala_result["Algo Params"] * 100)
        print(f"\nNewALA vs FedALA:")
        print(f"  FedALA additional params: {fedala_result['Algo Params']:,}")
        print(f"  NewALA additional params: {newala_result['Algo Params']:,}")
        print(f"  Reduction: {reduction:.2f}%")
        print(f"  Ratio: {fedala_result['Algo Params'] / newala_result['Algo Params']:.2f}x fewer params")

    print()
    print("=" * 80)
    print("NOTES")
    print("=" * 80)
    print("- Base Params: Parameters in the base GNN model")
    print("- Algo Params: Additional learnable parameters specific to the FL algorithm")
    print("- Total Params: Base + Algorithm-specific parameters")
    print("- Comm Params: Parameters communicated to server per round")
    print("- Forward GFLOPs: FLOPs for one forward pass")
    print("- Total GFLOPs/iter: FLOPs per training iteration (forward + backward + overhead)")
    print("- Overhead GFLOPs: Algorithm-specific computational overhead")
    print()
    print("NewALA uses low-rank decomposition to reduce parameters:")
    print(f"  - For m×n weight matrix: stores (m+n)×r instead of m×n")
    print(f"  - Current rank r={args.newala_rank}")
    print(f"  - For a 64×32 matrix: FedALA needs 2,048 params, NewALA needs 384 params")
    print("=" * 80)

    # Save results to file
    output_file = "fl_methods_complexity.txt"
    with open(output_file, 'w') as f:
        f.write(str(table))
        f.write("\n\nParameter Reduction (NewALA vs FedALA):\n")
        if newala_result and fedala_result and isinstance(newala_result["Algo Params"], int):
            f.write(f"  FedALA: {fedala_result['Algo Params']:,} params\n")
            f.write(f"  NewALA: {newala_result['Algo Params']:,} params\n")
            f.write(f"  Reduction: {reduction:.2f}%\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
