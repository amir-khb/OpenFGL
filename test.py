import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import traceback


def main():
    """
    Main function to run FedALA on all specified datasets.
    (Modified from original Table 7 recreation)
    """
    args = config.args

    # Configuration - matching paper specifications
    args.root = "/home/amirreza/ScalableProject/OpenFGL/dataset"  # CHANGE THIS to your data directory
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10
    args.task = "node_cls"
    args.metrics = ["accuracy"]
    args.num_rounds = 100
    args.num_epochs = 3
    args.lr = 1e-2
    args.weight_decay = 5e-4  # Added to match paper
    args.optimizer = 'adam'  # Added to match paper
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # --- MODIFICATION ---
    # Datasets from the original test.py
    datasets = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
        "ogbn-products",  # Products in table
        "Chameleon",
        "Actor",
        "Amazon-ratings"  # Ratings in table
    ]

    # --- MODIFICATION ---
    # Algorithm hardcoded to run FedALA only
    algorithms = {
        "FedALA": ("fedala", ["gcn"]),
    }
    # --- END MODIFICATION ---

    # Number of runs - matching paper (3 runs instead of 5)
    num_runs = 3

    # Storage for results
    results = {alg: {dataset: [] for dataset in datasets} for alg in algorithms.keys()}

    # Create output directory
    output_dir = f"fedala_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments
    total_experiments = len(algorithms) * len(datasets) * num_runs
    experiment_count = 0

    for alg_name, (alg, model) in algorithms.items():
        for dataset in datasets:
            print(f"\n{'=' * 80}")
            print(f"Running: {alg_name} on {dataset}")
            print(f"{'=' * 80}")

            # --- MODIFICATION ---
            # Removed OOM skip logic, as we only run FedALA
            # --- END MODIFICATION ---

            for run in range(num_runs):
                experiment_count += 1
                print(f"\nRun {run + 1}/{num_runs} - Progress: {experiment_count}/{total_experiments}")

                try:
                    # Set configuration
                    args.dataset = [dataset]
                    args.fl_algorithm = alg
                    args.model = model
                    args.seed = 2024 + run  # Different seed for each run

                    # Enable logging
                    args.debug = True
                    args.log_root = os.path.join(output_dir, "logs")
                    args.log_name = f"{alg_name}_{dataset}_run{run}"

                    # Run training
                    trainer = FGLTrainer(args)
                    trainer.train()

                    # Extract test accuracy from the trainer
                    test_acc = extract_test_accuracy(trainer)

                    if test_acc is not None:
                        results[alg_name][dataset].append(test_acc * 100)  # Convert to percentage
                        print(f"✓ Test Accuracy: {test_acc * 100:.2f}%")
                    else:
                        print(f"⚠ Warning: Could not extract test accuracy for {alg_name} on {dataset}")

                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Error running {alg_name} on {dataset}, run {run}: {error_msg}")
                    print(traceback.format_exc())

                    # Check for specific errors
                    if "out of memory" in error_msg.lower() or (
                            "cuda" in error_msg.lower() and "memory" in error_msg.lower()):
                        results[alg_name][dataset] = "OOM"
                        print(f"Marking as OOM and skipping remaining runs")
                        break
                    elif run == num_runs - 1:
                        # If all runs failed, mark as error
                        if not results[alg_name][dataset] or len(results[alg_name][dataset]) == 0:
                            results[alg_name][dataset] = "ERROR"

            # Save intermediate results after each dataset
            save_results(results, output_dir, "intermediate_results.json")

    # Compute statistics and create table
    create_table(results, datasets, algorithms, output_dir)

    print(f"\n{'=' * 80}")
    print(f"All experiments completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'=' * 80}")


def extract_test_accuracy(trainer):
    """
    Extract test accuracy from the trainer object.
    The correct key is 'best_test_accuracy' in trainer.evaluation_result
    """
    test_acc = None

    # Method 1: Check trainer.evaluation_result for best_test_accuracy
    if hasattr(trainer, 'evaluation_result') and isinstance(trainer.evaluation_result, dict):
        test_acc = trainer.evaluation_result.get('best_test_accuracy', None)
        if test_acc is not None:
            return test_acc

    # Method 2: Check logger.metrics_list for current_test_accuracy (last round)
    if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'metrics_list'):
        metrics_list = trainer.logger.metrics_list
        if metrics_list and len(metrics_list) > 0:
            # Get the best test accuracy across all rounds
            best_acc = None
            for metrics in metrics_list:
                if isinstance(metrics, dict):
                    current_acc = metrics.get('current_test_accuracy', None)
                    if current_acc is not None:
                        if best_acc is None or current_acc > best_acc:
                            best_acc = current_acc
            if best_acc is not None:
                return best_acc

    # Method 3: Try to get from saved log file
    if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_path'):
        try:
            import pickle
            log_path = trainer.logger.log_path
            if os.path.exists(log_path):
                with open(log_path, 'rb') as f:
                    log_data = pickle.load(f)
                    if 'metric' in log_data and log_data['metric']:
                        # Get best test accuracy from all metrics
                        best_acc = None
                        for metrics in log_data['metric']:
                            if isinstance(metrics, dict):
                                current_acc = metrics.get('current_test_accuracy', None)
                                if current_acc is not None:
                                    if best_acc is None or current_acc > best_acc:
                                        best_acc = current_acc
                        if best_acc is not None:
                            return best_acc
        except Exception as e:
            pass

    return test_acc


def save_results(results, output_dir, filename):
    """Save results to JSON file"""
    filepath = os.path.join(output_dir, filename)

    # Convert results to serializable format
    serializable_results = {}
    for alg in results:
        serializable_results[alg] = {}
        for dataset in results[alg]:
            val = results[alg][dataset]
            if isinstance(val, list):
                serializable_results[alg][dataset] = [float(x) for x in val]
            else:
                serializable_results[alg][dataset] = val

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_table(results, datasets, algorithms, output_dir):
    """Create formatted table with mean ± std"""

    # Prepare data for DataFrame
    table_data = {}

    for alg_name in algorithms.keys():
        row_data = []
        for dataset in datasets:
            result = results[alg_name][dataset]

            if result == "OOM":
                row_data.append("OOM")
            elif result == "ERROR" or (isinstance(result, str) and "ERROR" in result):
                row_data.append("ERROR")
            elif isinstance(result, list) and len(result) > 0:
                mean = np.mean(result)
                std = np.std(result)
                row_data.append(f"{mean:.1f}±{std:.1f}")
            else:
                row_data.append("N/A")

        table_data[alg_name] = row_data

    # Create DataFrame
    df = pd.DataFrame(table_data, index=datasets).T

    # Save to CSV
    csv_path = os.path.join(output_dir, "fedala_results.csv")
    df.to_csv(csv_path)
    print(f"\n✓ Table saved to: {csv_path}")

    # Print table
    print("\n" + "=" * 100)
    print("FedALA Test Accuracy (%)")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100)

    # Save formatted LaTeX table
    latex_path = os.path.join(output_dir, "fedala_latex.txt")
    with open(latex_path, 'w') as f:
        f.write(df.to_latex())
    print(f"✓ LaTeX table saved to: {latex_path}")

    # Find best and second-best for each dataset
    highlight_best(df, output_dir)


def highlight_best(df, output_dir):
    """Identify best and second-best results for each dataset"""

    best_results = {}

    for dataset in df.columns:
        values = []
        for alg in df.index:
            val = df.loc[alg, dataset]
            if val not in ["OOM", "N/A", "ERROR"] and isinstance(val, str) and '±' in val:
                try:
                    # Extract mean value
                    mean_val = float(val.split('±')[0])
                    values.append((alg, mean_val, val))
                except:
                    pass

        # Sort by accuracy (descending)
        values.sort(key=lambda x: x[1], reverse=True)

        if len(values) >= 2:
            best_results[dataset] = {
                'best': values[0],
                'second_best': values[1]
            }
        elif len(values) == 1:
            best_results[dataset] = {
                'best': values[0],
                'second_best': None
            }

    # Save best results
    best_path = os.path.join(output_dir, "best_results.txt")
    with open(best_path, 'w') as f:
        f.write("Best and Second-Best Results per Dataset:\n")
        f.write("=" * 80 + "\n")
        for dataset, res in best_results.items():
            f.write(f"\n{dataset}:\n")
            f.write(f"  Best: {res['best'][0]} - {res['best'][2]}\n")
            if res['second_best']:
                f.write(f"  Second: {res['second_best'][0]} - {res['second_best'][2]}\n")

    print(f"✓ Best results summary saved to: {best_path}")


if __name__ == "__main__":
    main()