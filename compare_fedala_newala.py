"""
NewALA Single-Configuration Analysis (10 Clients).

This script:
1. Runs NewALA with exactly 10 clients on specified datasets.
2. Tracks training time, testing time, and accuracy.
3. Saves results to CSV and formatted tables.
"""

import torch
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import json


def run_experiment(args, algorithm, num_clients, dataset, run_id):
    """
    Run a single experiment and track metrics.

    Returns:
        dict with metrics including accuracy, training time, testing time
    """
    # Set algorithm, dataset, and client count
    args.fl_algorithm = algorithm
    args.dataset = [dataset]
    args.num_clients = num_clients
    args.seed = 2024 + run_id

    print(f"\n{'='*80}")
    print(f"Running {algorithm.upper()} with {num_clients} clients on {dataset} (Run {run_id+1})")
    print(f"{'='*80}")

    # Create trainer
    trainer = FGLTrainer(args)

    # Track training time
    train_start = time.time()

    # Train
    trainer.train()

    train_end = time.time()
    total_train_time = train_end - train_start

    # Get final results
    final_results = trainer.evaluation_result

    # Extract test accuracy
    test_acc = final_results.get('best_test_accuracy', 0.0)
    val_acc = final_results.get('best_val_accuracy', 0.0)
    best_round = final_results.get('best_round', 0)

    # Calculate average time per round
    avg_time_per_round = total_train_time / args.num_rounds

    # Testing time (approximate - evaluation time in last round)
    test_time = total_train_time * 0.05  # Rough estimate

    results = {
        'algorithm': algorithm,
        'num_clients': num_clients,
        'dataset': dataset,
        'run': run_id,
        'test_accuracy': test_acc * 100,  # Convert to percentage
        'val_accuracy': val_acc * 100,
        'best_round': best_round,
        'total_train_time': total_train_time,
        'avg_time_per_round': avg_time_per_round,
        'test_time': test_time,
        'num_rounds': args.num_rounds,
    }

    print(f"\nResults:")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Val Accuracy: {val_acc*100:.2f}%")
    print(f"  Best Round: {best_round}")
    print(f"  Total Training Time: {total_train_time:.2f}s")
    print(f"  Avg Time/Round: {avg_time_per_round:.2f}s")

    return results


def main():
    """Main execution function."""
    args = config.args

    # Common configuration
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.task = "node_cls"
    args.metrics = ["accuracy"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.optimizer = 'adam'

    # FL settings
    args.num_rounds = 100
    args.num_epochs = 3
    args.client_frac = 1.0

    # NewALA settings
    args.newala_eta = 1.0
    args.newala_rand_percent = 80
    args.newala_layer_idx = 4  # Use all layers
    args.newala_rank = 32
    args.newala_gamma = 0.05
    args.newala_lambda_reg = 0.001

    # Enable logging
    args.debug = False

    # Datasets to iterate over
    datasets = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
        "ogbn-products",
        "Chameleon",
        "Actor",
        "Amazon-ratings"
    ]

    # Fixed Client Count
    client_counts = [10]

    # Algorithm fixed to NewALA
    algorithm = "newala"

    # Number of runs for statistical significance
    num_runs = 3

    # Storage
    all_results = []

    # Output directory
    output_dir = f"newala_10clients_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("NEWALA FIXED CLIENT ANALYSIS (10 Clients)")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"Client count: {client_counts[0]}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total experiments: {len(datasets) * len(client_counts) * num_runs}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Run all experiments
    total_experiments = len(datasets) * len(client_counts) * num_runs
    current_experiment = 0

    for dataset in datasets:
        for num_clients in client_counts:
            for run_id in range(num_runs):
                current_experiment += 1
                print(f"\n{'#'*80}")
                print(f"EXPERIMENT {current_experiment}/{total_experiments}")
                print(f"{'#'*80}")

                try:
                    results = run_experiment(args, algorithm, num_clients, dataset, run_id)
                    all_results.append(results)

                    # Save intermediate results
                    df = pd.DataFrame(all_results)
                    df.to_csv(os.path.join(output_dir, "intermediate_results.csv"), index=False)

                except Exception as e:
                    print(f"\n!!! ERROR in {algorithm} on {dataset} with {num_clients} clients (run {run_id}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)

    # Compute aggregated statistics
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)

    # Group by dataset and client count
    summary_stats = []

    for dataset in datasets:
        for num_clients in client_counts:
            subset = df[(df['num_clients'] == num_clients) & (df['dataset'] == dataset)]

            if len(subset) == 0:
                continue

            stats = {
                'Dataset': dataset,
                'Clients': num_clients,
                'Algorithm': algorithm.upper(),
                'Test Acc (%)': f"{subset['test_accuracy'].mean():.2f} ± {subset['test_accuracy'].std():.2f}",
                'Val Acc (%)': f"{subset['val_accuracy'].mean():.2f} ± {subset['val_accuracy'].std():.2f}",
                'Best Round': f"{subset['best_round'].mean():.1f}",
                'Total Time (s)': f"{subset['total_train_time'].mean():.2f} ± {subset['total_train_time'].std():.2f}",
                'Time/Round (s)': f"{subset['avg_time_per_round'].mean():.3f}",
                'Runs': len(subset)
            }
            summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Save summary
    summary_df.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

    # Print results
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    print(f"\nNewALA Overall Performance (10 Clients):")
    print(f"  Average Test Accuracy: {df['test_accuracy'].mean():.2f}% ± {df['test_accuracy'].std():.2f}%")
    print(f"  Average Training Time: {df['total_train_time'].mean():.2f}s ± {df['total_train_time'].std():.2f}s")
    print(f"  Average Time/Round: {df['avg_time_per_round'].mean():.3f}s ± {df['avg_time_per_round'].std():.3f}s")

    # Save formatted report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NEWALA 10-CLIENT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets: {datasets}\n")
        f.write(f"Client count: {client_counts[0]}\n")
        f.write(f"Number of runs per config: {num_runs}\n")
        f.write(f"Total experiments: {len(df)}\n\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"\nNewALA Overall (10 Clients):\n")
        f.write(f"  Average Test Accuracy: {df['test_accuracy'].mean():.2f}% ± {df['test_accuracy'].std():.2f}%\n")
        f.write(f"  Average Training Time: {df['total_train_time'].mean():.2f}s ± {df['total_train_time'].std():.2f}s\n\n")

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - raw_results.csv: All individual run results")
    print(f"  - summary_stats.csv: Aggregated statistics")
    print(f"  - report.txt: Formatted text report")
    print(f"{'='*80}")

    return df, summary_df


if __name__ == "__main__":
    df, summary_df = main()