"""
Example script to test the NewALA (LoRA-CAAA) implementation.

This script demonstrates how to run the NewALA algorithm on a sample dataset.
"""

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


def main():
    """
    Main function to run NewALA on a sample dataset.
    """
    args = config.args

    # Basic configuration
    args.root = "./dataset"  # Change this to your data directory
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 5
    args.task = "node_cls"
    args.metrics = ["accuracy"]
    args.num_rounds = 50
    args.num_epochs = 3
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.optimizer = 'adam'
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # Dataset
    args.dataset = ["Cora"]  # Start with a small dataset for testing

    # Model
    args.model = ["gcn"]

    # NewALA algorithm
    args.fl_algorithm = "newala"

    # NewALA-specific parameters (LoRA-CAAA)
    args.newala_eta = 1.0  # Learning rate for low-rank matrix optimization
    args.newala_rand_percent = 80  # Percentage of training data to sample
    args.newala_layer_idx = 2  # Number of layers to apply NewALA (from top)
    args.newala_rank = 4  # Rank for low-rank decomposition
    args.newala_gamma = 0.1  # Sensitivity for confidence-aware gating
    args.newala_lambda_reg = 0.01  # Regularization weight for entropy term

    # Seed for reproducibility
    args.seed = 2024

    # Enable debug mode
    args.debug = True

    print("=" * 80)
    print("Running NewALA (LoRA-CAAA) on Cora dataset")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset[0]}")
    print(f"  - Model: {args.model[0]}")
    print(f"  - FL Algorithm: {args.fl_algorithm}")
    print(f"  - Num Clients: {args.num_clients}")
    print(f"  - Num Rounds: {args.num_rounds}")
    print(f"  - Local Epochs: {args.num_epochs}")
    print(f"\nNewALA Parameters:")
    print(f"  - Rank (r): {args.newala_rank}")
    print(f"  - Gamma (γ): {args.newala_gamma}")
    print(f"  - Lambda (λ): {args.newala_lambda_reg}")
    print(f"  - Eta (η): {args.newala_eta}")
    print("=" * 80)

    # Run training
    trainer = FGLTrainer(args)
    trainer.train()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
