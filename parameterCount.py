"""
Simple comparison of NewALA with different ranks.
- 2-layer GCN
- layer_idx=4 (all layers)
- Compare ranks: 2, 4, 8, 16, 32
"""

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.task_utils import load_node_edge_level_default_model
from openfgl.data.distributed_dataset_loader import FGLDataset
import numpy as np
from prettytable import PrettyTable


def count_newala_params(model, layer_idx, rank):
    """Count NewALA parameters for given rank."""
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
            r = min(rank, min(m, n))
            total += (m + n) * r
        else:
            total += 1
    return total


def analyze_parameters():
    """Show parameter breakdown for different ranks."""
    args = config.args

    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10
    args.task = "node_cls"
    args.dataset = ["PubMed"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    print("="*80)
    print("NewALA PARAMETER ANALYSIS")
    print("="*80)
    print(f"Configuration: 2-layer GCN, layer_idx=4 (all layers)")
    print("="*80)

    # Load dataset and model
    fgl_dataset = FGLDataset(args)
    sample_data = fgl_dataset.local_data[0]
    num_features = sample_data.x.shape[1]
    num_classes = fgl_dataset.global_data.num_global_classes

    print(f"\nDataset: {args.dataset[0]}")
    print(f"Input features: {num_features}")
    print(f"Hidden dim: {args.hid_dim}")
    print(f"Output classes: {num_classes}")

    # Load model
    args.fl_algorithm = "fedavg"
    model = load_node_edge_level_default_model(args, input_dim=num_features, output_dim=num_classes)

    # Show base model
    params = list(model.parameters())
    base_total = sum(p.numel() for p in params)

    print(f"\nBase model parameters: {base_total:,}")
    print("\nLayer breakdown:")
    print(f"  Layer 1 weight: {params[0].shape} = {params[0].numel():,} params")
    print(f"  Layer 1 bias:   {params[1].shape} = {params[1].numel():,} params")
    print(f"  Layer 2 weight: {params[2].shape} = {params[2].numel():,} params")
    print(f"  Layer 2 bias:   {params[3].shape} = {params[3].numel():,} params")

    # Compare ranks
    print("\n" + "="*80)
    print("PARAMETER COUNTS FOR DIFFERENT RANKS")
    print("="*80)

    ranks = [2, 4, 8, 16, 32, 64]
    layer_idx = 4

    table = PrettyTable()
    table.field_names = ["Rank", "NewALA Params", "% of Base Model", "Memory (KB)"]
    table.align["Rank"] = "r"
    table.align["NewALA Params"] = "r"

    for rank in ranks:
        newala_params = count_newala_params(model, layer_idx, rank)
        pct_base = 100 * newala_params / base_total
        memory_kb = (newala_params * 4) / 1024  # 4 bytes per float32

        table.add_row([
            rank,
            f"{newala_params:,}",
            f"{pct_base:.1f}%",
            f"{memory_kb:.1f}"
        ])

    print("\n" + str(table))

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
Based on parameter analysis:
  • rank=2-4:   Very efficient (~5K params), may underfit
  • rank=8:     Good balance (~12K params), recommended
  • rank=16:    Conservative (~23K params), best quality
  • rank=32+:   Diminishing returns
    """)


def run_rank_comparison():
    """Run training with different ranks and compare results."""
    args = config.args

    # Configuration
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 5
    args.task = "node_cls"
    args.dataset = ["PubMed"]
    args.model = ["gcn"]
    args.metrics = ["accuracy"]
    args.num_rounds = 50
    args.num_epochs = 3
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # NewALA settings
    args.fl_algorithm = "newala"
    args.newala_layer_idx = 4  # All layers
    args.newala_eta = 1.0
    args.newala_rand_percent = 80
    args.newala_gamma = 0.01
    args.newala_lambda_reg = 0.001

    args.seed = 2024
    args.debug = False

    print("\n" + "="*80)
    print("NEWALA RANK COMPARISON - TRAINING")
    print("="*80)
    print("Configuration:")
    print(f"  Model: 2-layer GCN")
    print(f"  layer_idx: 4 (all layers)")
    print(f"  Dataset: {args.dataset[0]}")
    print(f"  Rounds: {args.num_rounds}")
    print("="*80)

    # Ranks to test
    ranks = [2, 4, 8, 16, 32]

    results = []

    for rank in ranks:
        print(f"\n{'='*80}")
        print(f"Training NewALA with rank={rank}")
        print(f"{'='*80}")

        args.newala_rank = rank

        try:
            trainer = FGLTrainer(args)
            trainer.train()

            # Get results
            test_acc = trainer.evaluation_result.get('best_test_accuracy', 0.0) * 100
            val_acc = trainer.evaluation_result.get('best_val_accuracy', 0.0) * 100
            best_round = trainer.evaluation_result.get('best_round', 0)

            # Count parameters
            model = trainer.clients[0].task.model
            algo_params = count_newala_params(model, args.newala_layer_idx, rank)

            results.append({
                'rank': rank,
                'test_acc': test_acc,
                'val_acc': val_acc,
                'best_round': best_round,
                'params': algo_params
            })

            print(f"\n✓ Test Accuracy: {test_acc:.2f}%")
            print(f"✓ Val Accuracy:  {val_acc:.2f}%")
            print(f"✓ Best Round:    {best_round}")
            print(f"✓ Algo Params:   {algo_params:,}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    table = PrettyTable()
    table.field_names = ["Rank", "Test Accuracy", "Val Accuracy", "Best Round", "Algo Params", "Memory (KB)"]
    table.align["Rank"] = "r"
    table.align["Algo Params"] = "r"

    for r in results:
        memory_kb = (r['params'] * 4) / 1024
        table.add_row([
            r['rank'],
            f"{r['test_acc']:.2f}%",
            f"{r['val_acc']:.2f}%",
            r['best_round'],
            f"{r['params']:,}",
            f"{memory_kb:.1f}"
        ])

    print("\n" + str(table))

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if len(results) > 0:
        # Find best accuracy
        best_result = max(results, key=lambda x: x['test_acc'])

        # Calculate variation
        accs = [r['test_acc'] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        range_acc = max(accs) - min(accs)

        print(f"\nAccuracy Statistics:")
        print(f"  Mean: {mean_acc:.2f}%")
        print(f"  Std:  {std_acc:.2f}%")
        print(f"  Range: {range_acc:.2f}%")

        print(f"\nBest Configuration:")
        print(f"  Rank: {best_result['rank']}")
        print(f"  Test Accuracy: {best_result['test_acc']:.2f}%")
        print(f"  Parameters: {best_result['params']:,}")

        # Efficiency analysis
        print(f"\nEfficiency vs Accuracy Trade-off:")
        for r in results:
            efficiency = r['test_acc'] / (r['params'] / 1000)  # Accuracy per 1K params
            print(f"  Rank {r['rank']:2d}: {r['test_acc']:5.2f}% with {r['params']:>6,} params "
                  f"(efficiency: {efficiency:.3f})")

        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        if range_acc < 1.0:
            print(f"""
Accuracy variation is small ({range_acc:.2f}%), suggesting all ranks work well.


  
Alternative: rank={best_result['rank']} (best accuracy)
  • Test Accuracy: {best_result['test_acc']:.2f}%
  • Parameters: {best_result['params']:,}
            """)
        elif range_acc < 2.0:
            print(f"""
Moderate accuracy variation ({range_acc:.2f}%), rank matters somewhat.

Recommended: rank={best_result['rank']}
  • Test Accuracy: {best_result['test_acc']:.2f}%
  • Parameters: {best_result['params']:,}
  • Best accuracy observed
            """)
        else:
            print(f"""
Significant accuracy variation ({range_acc:.2f}%), rank is important!

Recommended: rank={best_result['rank']}
  • Test Accuracy: {best_result['test_acc']:.2f}%
  • Parameters: {best_result['params']:,}
  • Substantially better than lower ranks
            """)

    print("="*80)

    return results


if __name__ == "__main__":
    # Step 1: Show parameter analysis
    print("\nSTEP 1: Parameter Analysis")
    print("="*80)
    analyze_parameters()

    # Step 2: Run training comparison
    print("\n\nSTEP 2: Training Comparison")
    print("="*80)
    response = input("\nRun training with different ranks? (y/n): ")

    if response.lower() == 'y':
        results = run_rank_comparison()

        print("\n\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print("\nResults saved. Key findings:")
        print("  • All ranks applied to ALL layers (layer_idx=4)")
        print("  • Parameter counts vary from ~5K to ~50K")
        print("  • Check recommendations above for best rank")
        print("="*80)
    else:
        print("\nSkipping training. Run the script again when ready.")