"""
Script to run FedALA (Adaptive Local Aggregation) on OpenFGL framework
"""
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


def main():
    """
    Main function to run FedALA experiments
    """
    args = config.args

    # ============ Basic Configuration ============
    args.root = "/home/amirreza/ScalableProject/AmirOpenFGL/OpenFGL/dataset"  # CHANGE THIS to your data directory
    args.scenario = "subgraph_fl"
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10
    args.task = "node_cls"
    args.metrics = ["accuracy"]

    # ============ Training Configuration ============
    args.num_rounds = 100
    args.num_epochs = 3
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.optimizer = 'adam'
    args.batch_size = 128

    # ============ Model Configuration ============
    args.model = ["gcn"]  # Can use: gcn, gat, graphsage, sgc, etc.
    args.hid_dim = 64
    args.num_layers = 2
    args.dropout = 0.5

    # ============ FedALA-Specific Configuration ============
    args.fl_algorithm = "fedala"
    args.ala_eta = 1.0  # Learning rate for weight learning
    args.ala_rand_percent = 80  # Percentage of training data to sample
    args.ala_layer_idx = 2  # Number of top layers to apply ALA

    # ============ Dataset Configuration ============
    # Choose from: Cora, CiteSeer, PubMed, Photo, Computers, etc.
    args.dataset = ["Cora"]

    # ============ Logging ============
    args.debug = True
    args.log_root = "./logs"
    args.log_name = f"fedala_{args.dataset[0]}"

    # ============ Run Training ============
    print("=" * 80)
    print(f"Running FedALA on {args.dataset[0]}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"ALA eta: {args.ala_eta}")
    print(f"ALA rand_percent: {args.ala_rand_percent}")
    print(f"ALA layer_idx: {args.ala_layer_idx}")
    print("=" * 80)

    trainer = FGLTrainer(args)
    trainer.train()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()