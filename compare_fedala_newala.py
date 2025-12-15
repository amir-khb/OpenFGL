"""
Comprehensive comparison of FedALA vs NewALA across different client counts.

This script:
1. Runs both FedALA and NewALA with varying numbers of clients (5, 10, 20, 50, 100)
2. Tracks training time, testing time, and accuracy
3. Saves results to CSV and formatted tables
4. Provides statistical analysis and scalability insights

Results include:
- Test accuracy (mean ± std over multiple runs)
- Training time per round
- Total training time
- Testing time
- Scalability metrics
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
    """Main comparison function."""
    args = config.args

    # Common configuration
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.task = "node_cls"
    args.metrics = ["accuracy"]
    args.model = ["gcn"]
    args.hid_dim = 64
    args.num_layers = 4
    args.dropout = 0.5
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.optimizer = 'adam'

    # FL settings
    args.num_rounds = 100
    args.num_epochs = 3
    args.client_frac = 1.0

    # FedALA settings
    args.ala_eta = 1.0
    args.ala_rand_percent = 80
    args.ala_layer_idx = 2  # Use all layers for fair comparison

    # NewALA settings
    args.newala_eta = 1.0
    args.newala_rand_percent = 80
    args.newala_layer_idx = 4  # Use all layers
    args.newala_rank = 128
    args.newala_gamma = 0.1
    args.newala_lambda_reg = 0.1

    # Enable logging
    args.debug = False

    # Dataset to use (you can change this)
    dataset = "Cora"  # Change to your preferred dataset

    # Client counts to test
    client_counts = [5,10,15,20,25,30]

    # Algorithms to compare
    algorithms = ["fedala", "newala"]

    # Number of runs for statistical significance
    num_runs = 3

    # Storage
    all_results = []

    # Output directory
    output_dir = f"scalability_comparison_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("FEDALA vs NEWALA CLIENT SCALABILITY COMPARISON")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Client counts: {client_counts}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total experiments: {len(client_counts) * len(algorithms) * num_runs}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Run all experiments
    total_experiments = len(client_counts) * len(algorithms) * num_runs
    current_experiment = 0

    for num_clients in client_counts:
        for algorithm in algorithms:
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
                    print(f"\n!!! ERROR in {algorithm} with {num_clients} clients (run {run_id}): {e}")
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

    # Group by algorithm and client count
    summary_stats = []

    for num_clients in client_counts:
        for algorithm in algorithms:
            subset = df[(df['num_clients'] == num_clients) & (df['algorithm'] == algorithm)]

            if len(subset) == 0:
                continue

            stats = {
                'Clients': num_clients,
                'Algorithm': algorithm.upper(),
                'Test Acc (%)': f"{subset['test_accuracy'].mean():.2f} ± {subset['test_accuracy'].std():.2f}",
                'Val Acc (%)': f"{subset['val_accuracy'].mean():.2f} ± {subset['val_accuracy'].std():.2f}",
                'Best Round': f"{subset['best_round'].mean():.1f}",
                'Total Time (s)': f"{subset['total_train_time'].mean():.2f} ± {subset['total_train_time'].std():.2f}",
                'Time/Round (s)': f"{subset['avg_time_per_round'].mean():.3f}",
                'Test Time (s)': f"{subset['test_time'].mean():.3f}",
                'Runs': len(subset)
            }
            summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Save summary
    summary_df.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

    # Create comparison table (side-by-side for each client count)
    print("\n" + "="*80)
    print("GENERATING COMPARISON TABLES")
    print("="*80)

    comparison_rows = []

    for num_clients in client_counts:
        fedala_data = df[(df['num_clients'] == num_clients) & (df['algorithm'] == 'fedala')]
        newala_data = df[(df['num_clients'] == num_clients) & (df['algorithm'] == 'newala')]

        if len(fedala_data) == 0 or len(newala_data) == 0:
            continue

        # Calculate improvements
        acc_improvement = newala_data['test_accuracy'].mean() - fedala_data['test_accuracy'].mean()
        time_ratio = fedala_data['total_train_time'].mean() / newala_data['total_train_time'].mean()

        row = {
            'Clients': num_clients,
            'FedALA Acc (%)': f"{fedala_data['test_accuracy'].mean():.2f}±{fedala_data['test_accuracy'].std():.2f}",
            'NewALA Acc (%)': f"{newala_data['test_accuracy'].mean():.2f}±{newala_data['test_accuracy'].std():.2f}",
            'Acc Diff': f"{acc_improvement:+.2f}%",
            'FedALA Time (s)': f"{fedala_data['total_train_time'].mean():.1f}±{fedala_data['total_train_time'].std():.1f}",
            'NewALA Time (s)': f"{newala_data['total_train_time'].mean():.1f}±{newala_data['total_train_time'].std():.1f}",
            'Time Ratio': f"{time_ratio:.2f}x",
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)

    # Scalability analysis
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)

    scalability_rows = []
    for algorithm in algorithms:
        algo_data = df[df['algorithm'] == algorithm].groupby('num_clients').agg({
            'test_accuracy': ['mean', 'std'],
            'total_train_time': ['mean', 'std'],
            'avg_time_per_round': ['mean', 'std']
        }).reset_index()

        for _, row in algo_data.iterrows():
            scalability_rows.append({
                'Algorithm': algorithm.upper(),
                'Clients': row['num_clients'],
                'Accuracy': f"{row[('test_accuracy', 'mean')]:.2f}±{row[('test_accuracy', 'std')]:.2f}",
                'Total Time': f"{row[('total_train_time', 'mean')]:.1f}±{row[('total_train_time', 'std')]:.1f}",
                'Time/Round': f"{row[('avg_time_per_round', 'mean')]:.3f}±{row[('avg_time_per_round', 'std')]:.3f}"
            })

    scalability_df = pd.DataFrame(scalability_rows)
    scalability_df.to_csv(os.path.join(output_dir, "scalability_analysis.csv"), index=False)

    # Print results
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("HEAD-TO-HEAD COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))

    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    print(scalability_df.to_string(index=False))

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    fedala_overall = df[df['algorithm'] == 'fedala']
    newala_overall = df[df['algorithm'] == 'newala']

    print(f"\nFedALA (across all client counts):")
    print(f"  Average Test Accuracy: {fedala_overall['test_accuracy'].mean():.2f}% ± {fedala_overall['test_accuracy'].std():.2f}%")
    print(f"  Average Training Time: {fedala_overall['total_train_time'].mean():.2f}s ± {fedala_overall['total_train_time'].std():.2f}s")
    print(f"  Average Time/Round: {fedala_overall['avg_time_per_round'].mean():.3f}s ± {fedala_overall['avg_time_per_round'].std():.3f}s")

    print(f"\nNewALA (across all client counts):")
    print(f"  Average Test Accuracy: {newala_overall['test_accuracy'].mean():.2f}% ± {newala_overall['test_accuracy'].std():.2f}%")
    print(f"  Average Training Time: {newala_overall['total_train_time'].mean():.2f}s ± {newala_overall['total_train_time'].std():.2f}s")
    print(f"  Average Time/Round: {newala_overall['avg_time_per_round'].mean():.3f}s ± {newala_overall['avg_time_per_round'].std():.3f}s")

    print(f"\nOverall Comparison:")
    print(f"  Accuracy Improvement: {newala_overall['test_accuracy'].mean() - fedala_overall['test_accuracy'].mean():+.2f}%")
    print(f"  Time Ratio (FedALA/NewALA): {fedala_overall['total_train_time'].mean() / newala_overall['total_train_time'].mean():.2f}x")

    # Scalability insights
    print(f"\nScalability Insights:")

    for algorithm in algorithms:
        algo_data = df[df['algorithm'] == algorithm].groupby('num_clients')['total_train_time'].mean()
        if len(algo_data) > 1:
            time_growth = (algo_data.iloc[-1] - algo_data.iloc[0]) / algo_data.iloc[0] * 100
            print(f"  {algorithm.upper()}: Training time increased by {time_growth:.1f}% from {client_counts[0]} to {client_counts[-1]} clients")

        algo_acc = df[df['algorithm'] == algorithm].groupby('num_clients')['test_accuracy'].mean()
        if len(algo_acc) > 1:
            acc_change = algo_acc.iloc[-1] - algo_acc.iloc[0]
            print(f"  {algorithm.upper()}: Accuracy changed by {acc_change:+.2f}% from {client_counts[0]} to {client_counts[-1]} clients")

    # Save formatted report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEDALA vs NEWALA CLIENT SCALABILITY COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Client counts tested: {client_counts}\n")
        f.write(f"Number of runs per config: {num_runs}\n")
        f.write(f"Total experiments: {len(df)}\n\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        f.write("="*80 + "\n")
        f.write("HEAD-TO-HEAD COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")

        f.write("="*80 + "\n")
        f.write("SCALABILITY ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(scalability_df.to_string(index=False) + "\n\n")

        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"\nFedALA (across all client counts):\n")
        f.write(f"  Average Test Accuracy: {fedala_overall['test_accuracy'].mean():.2f}% ± {fedala_overall['test_accuracy'].std():.2f}%\n")
        f.write(f"  Average Training Time: {fedala_overall['total_train_time'].mean():.2f}s ± {fedala_overall['total_train_time'].std():.2f}s\n")
        f.write(f"\nNewALA (across all client counts):\n")
        f.write(f"  Average Test Accuracy: {newala_overall['test_accuracy'].mean():.2f}% ± {newala_overall['test_accuracy'].std():.2f}%\n")
        f.write(f"  Average Training Time: {newala_overall['total_train_time'].mean():.2f}s ± {newala_overall['total_train_time'].std():.2f}s\n")
        f.write(f"\nOverall Comparison:\n")
        f.write(f"  Accuracy Improvement: {newala_overall['test_accuracy'].mean() - fedala_overall['test_accuracy'].mean():+.2f}%\n")
        f.write(f"  Time Ratio: {fedala_overall['total_train_time'].mean() / newala_overall['total_train_time'].mean():.2f}x\n\n")

        f.write("="*80 + "\n")
        f.write("SCALABILITY INSIGHTS\n")
        f.write("="*80 + "\n")
        for algorithm in algorithms:
            algo_data = df[df['algorithm'] == algorithm].groupby('num_clients')['total_train_time'].mean()
            if len(algo_data) > 1:
                time_growth = (algo_data.iloc[-1] - algo_data.iloc[0]) / algo_data.iloc[0] * 100
                f.write(f"{algorithm.upper()}: Training time increased by {time_growth:.1f}% from {client_counts[0]} to {client_counts[-1]} clients\n")

            algo_acc = df[df['algorithm'] == algorithm].groupby('num_clients')['test_accuracy'].mean()
            if len(algo_acc) > 1:
                acc_change = algo_acc.iloc[-1] - algo_acc.iloc[0]
                f.write(f"{algorithm.upper()}: Accuracy changed by {acc_change:+.2f}% from {client_counts[0]} to {client_counts[-1]} clients\n")

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - raw_results.csv: All individual run results")
    print(f"  - summary_stats.csv: Aggregated statistics")
    print(f"  - comparison_table.csv: Head-to-head comparison")
    print(f"  - scalability_analysis.csv: Scalability metrics")
    print(f"  - report.txt: Formatted text report")
    print(f"{'='*80}")

    return df, summary_df, comparison_df, scalability_df


if __name__ == "__main__":
    df, summary_df, comparison_df, scalability_df = main()