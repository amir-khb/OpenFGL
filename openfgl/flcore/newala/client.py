import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from typing import List, Tuple

from openfgl.flcore.base import BaseClient


class LoRA_CAAA:
    """
    LoRA-CAAA: Low-Rank Confidence-Aware Adaptive Aggregation

    This class implements the unified formulation combining:
    1. Low-Rank Parameterization (solving parameter explosion)
    2. Confidence-Aware Gating (solving noise sensitivity)
    3. Entropy Regularization (solving binary hardness)
    """

    def __init__(self,
                 cid: int,
                 loss_fn: nn.Module,
                 data_obj: 'torch_geometric.data.Data',
                 train_mask: torch.Tensor,
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 threshold: float = 0.1,
                 num_pre_loss: int = 10,
                 rank: int = 4,
                 gamma: float = 0.1,
                 lambda_reg: float = 0.01) -> None:
        """
        Initialize the LoRA-CAAA module for GNN-compatible federated learning.

        Args:
            cid: Client ID.
            loss_fn: The loss function (e.g., nn.CrossEntropyLoss).
            data_obj: The PyG Data object for this client.
            train_mask: The train mask for the PyG Data object.
            rand_percent: Percent of local training data to sample for weight learning.
            layer_idx: Control the weight range (from top layers).
            eta: Learning rate for low-rank matrix optimization.
            device: Using cuda or cpu.
            threshold: Convergence threshold for weight learning.
            num_pre_loss: Window size for checking convergence.
            rank: Rank for low-rank decomposition (r << min(m,n)).
            gamma: Sensitivity hyperparameter for confidence-aware gating.
            lambda_reg: Regularization weight for entropy regularization.
        """

        self.cid = cid
        self.loss = loss_fn
        self.data_obj = data_obj
        self.train_mask = train_mask
        self.train_indices = train_mask.nonzero(as_tuple=False).view(-1)

        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        # LoRA-CAAA specific parameters
        self.rank = rank
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        # Low-rank matrices A and B for each parameter
        self.A_matrices = None  # List of A matrices
        self.B_matrices = None  # List of B matrices
        self.start_phase = True

    def compute_entropy(self, logits: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Compute the batch entropy E(x) from model predictions.

        E(x) = -1/N * Σ Σ p_k(x_i) log p_k(x_i)

        Args:
            logits: Model output logits [num_nodes, num_classes]
            mask: Mask indicating which nodes to consider

        Returns:
            Entropy value E(x)
        """
        # Get probability distribution
        probs = F.softmax(logits[mask], dim=1)

        # Compute entropy: -Σ p log p
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)

        # Normalize by number of samples and classes
        entropy = entropy / mask.sum().item()

        return entropy.item()

    def compute_trust_coefficient(self, entropy: float) -> float:
        """
        Compute the trust coefficient β based on model confidence.

        β = exp(-γ · E(x))

        When entropy is high (model is uncertain), β → 0, suppressing the update.
        When entropy is low (model is confident), β → 1, allowing the update.

        Args:
            entropy: Batch entropy E(x)

        Returns:
            Trust coefficient β ∈ (0, 1]
        """
        beta = np.exp(-self.gamma * entropy)
        return beta

    def compute_entropy_regularization(self, W_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute entropy regularization to prevent W from collapsing to binary values.

        R(W) = -Σ [W_ij log W_ij + (1 - W_ij) log(1 - W_ij)]

        Args:
            W_list: List of weight matrices (aggregation weights)

        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        epsilon = 1e-10  # To avoid log(0)

        for W in W_list:
            # Clamp W to valid range for log
            W_clamped = torch.clamp(W, epsilon, 1 - epsilon)

            # Binary cross-entropy style regularization
            reg = -(W_clamped * torch.log(W_clamped) +
                   (1 - W_clamped) * torch.log(1 - W_clamped))

            reg_loss += reg.sum()

        return reg_loss

    def low_rank_decomposition(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize low-rank matrices A and B such that W ≈ σ(A · B^T)

        For W ∈ R^(m×n), we create:
        - A ∈ R^(n×r)
        - B ∈ R^(m×r)
        where r << min(m, n)

        Args:
            W: Weight matrix to decompose

        Returns:
            Tuple of (A, B) matrices
        """
        m, n = W.shape  # Note: W shape is actually the parameter shape
        r = min(self.rank, min(m, n))

        # Use SVD to initialize A and B
        # W = U S V^T, we can set A = V S^(1/2), B = U S^(1/2)
        # Then A B^T = V S^(1/2) S^(1/2) U^T = V S U^T ≈ W (after taking top r components)

        with torch.no_grad():
            # Apply inverse sigmoid to W to initialize in the right space
            # since W = σ(A·B^T), we want A·B^T ≈ σ^(-1)(W)
            W_clamped = torch.clamp(W, 0.01, 0.99)
            W_logit = torch.log(W_clamped / (1 - W_clamped))  # Inverse sigmoid

            try:
                U, S, Vt = torch.linalg.svd(W_logit, full_matrices=False)

                # Take top r singular values
                S_r = S[:r]
                U_r = U[:, :r]
                V_r = Vt[:r, :].T

                # Initialize A and B
                sqrt_S = torch.sqrt(S_r)
                A = V_r * sqrt_S.unsqueeze(0)  # (n, r)
                B = U_r * sqrt_S.unsqueeze(0)  # (m, r)
            except:
                # If SVD fails, use random initialization
                A = torch.randn(n, r, device=self.device) * 0.01
                B = torch.randn(m, r, device=self.device) * 0.01

        A.requires_grad = True
        B.requires_grad = True

        return A, B

    def adaptive_local_aggregation(self,
                                   global_model: nn.Module,
                                   local_model: nn.Module) -> None:
        """
        Performs adaptive local aggregation using the LoRA-CAAA framework.

        The unified algorithm:
        1. Compute W = σ(A · B^T) (low-rank parameterization)
        2. Compute β = exp(-γ · E(x)) (confidence-aware gating)
        3. Initialize: θ_init = θ_l + β · [(θ_g - θ_l) ⊙ W]
        4. Optimize: min_{A,B} (L_task(θ_init) - λ R(W))

        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """

        # Get random subset for weight learning
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_indices))

        # Obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # Deactivate LoRA-CAAA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # Preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # Create temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # Only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # Freeze the lower layers to reduce computational cost
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # Initialize low-rank matrices A and B for each parameter
        if self.A_matrices is None:
            self.A_matrices = []
            self.B_matrices = []

            for param in params_p:
                # Check if parameter is 2D (weight matrix) or 1D (bias vector)
                if param.dim() >= 2:
                    # For 2D+ parameters, use low-rank decomposition
                    W_init = torch.ones_like(param.data).to(self.device)
                    A, B = self.low_rank_decomposition(W_init)
                    self.A_matrices.append(A)
                    self.B_matrices.append(B)
                else:
                    # For 1D parameters (biases), use scalar weights instead
                    # Store None to indicate this parameter uses simple weighting
                    self.A_matrices.append(None)
                    self.B_matrices.append(None)

        # Create optimizer for A and B matrices (filter out None values for 1D parameters)
        all_matrices = [m for m in (self.A_matrices + self.B_matrices) if m is not None]
        optimizer = torch.optim.SGD(all_matrices, lr=self.eta) if all_matrices else None

        # Ensure data is on the correct device
        self.data_obj = self.data_obj.to(self.device)

        # Weight learning loop
        losses = []
        cnt = 0

        while True:
            # Create a random mask for this iteration
            shuffled_indices = self.train_indices[torch.randperm(len(self.train_indices))]
            rand_indices = shuffled_indices[:rand_num]
            rand_mask = torch.zeros_like(self.train_mask, dtype=torch.bool)
            rand_mask[rand_indices] = True

            # Forward pass to compute entropy for confidence-aware gating
            with torch.no_grad():
                _, output = model_t(self.data_obj)
                entropy = self.compute_entropy(output, rand_mask)
                beta = self.compute_trust_coefficient(entropy)

            # Compute aggregation weights W = σ(A · B^T) and apply trust coefficient
            W_list = []
            for param, A, B in zip(params_p, self.A_matrices, self.B_matrices):
                if A is not None and B is not None:
                    # For 2D parameters: W = σ(A · B^T)
                    W = torch.sigmoid(torch.matmul(A, B.T))
                else:
                    # For 1D parameters: use uniform weights (fully trust the aggregation)
                    W = torch.ones_like(param.data).to(self.device)
                W_list.append(W)

            # Update temp model parameters: θ_init = θ_l + β · [(θ_g - θ_l) ⊙ W]
            for param_t, param, param_g, W in zip(params_tp, params_p, params_gp, W_list):
                if param.dim() >= 2:
                    param_t.data = param.data + beta * ((param_g.data - param.data) * W.T)
                else:
                    # For 1D parameters, W is already the right shape
                    param_t.data = param.data + beta * ((param_g.data - param.data) * W)

            # Compute task loss
            if optimizer is not None:
                optimizer.zero_grad()
            _, output = model_t(self.data_obj)
            task_loss = self.loss(output[rand_mask], self.data_obj.y[rand_mask])

            # Compute entropy regularization (only for 2D parameters with A, B matrices)
            W_list_2d = [W for W, A in zip(W_list, self.A_matrices) if A is not None]
            reg_loss = self.compute_entropy_regularization(W_list_2d) if W_list_2d else 0.0

            # Total loss: L_task - λ R(W)
            # Note: We want to maximize R(W), so we subtract it
            if isinstance(reg_loss, torch.Tensor):
                total_loss = task_loss - self.lambda_reg * reg_loss
            else:
                total_loss = task_loss

            total_loss.backward()

            # Update A and B matrices
            if optimizer is not None:
                optimizer.step()

            losses.append(task_loss.item())
            cnt += 1

            # Only train one epoch in subsequent iterations
            if not self.start_phase:
                break

            # Train until convergence in the first phase
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client: {self.cid}\tStd: {np.std(losses[-self.num_pre_loss:]):.4f}\t'
                      f'LoRA-CAAA epochs: {cnt}\tBeta: {beta:.4f}')
                break

        self.start_phase = False

        # Apply final aggregation weights to obtain initialized local model
        with torch.no_grad():
            # Recompute final W and beta
            _, output = model_t(self.data_obj)
            entropy = self.compute_entropy(output, self.train_mask)
            beta = self.compute_trust_coefficient(entropy)

            for param, param_g, A, B in zip(params_p, params_gp, self.A_matrices, self.B_matrices):
                if A is not None and B is not None:
                    # For 2D parameters: W = σ(A · B^T)
                    W = torch.sigmoid(torch.matmul(A, B.T))
                    param.data = param.data + beta * ((param_g.data - param.data) * W.T)
                else:
                    # For 1D parameters: use uniform weights
                    W = torch.ones_like(param.data).to(self.device)
                    param.data = param.data + beta * ((param_g.data - param.data) * W)


class NewALAClient(BaseClient):
    """
    NewALAClient implements the client-side logic for the LoRA-CAAA algorithm.

    LoRA-CAAA combines:
    - Low-Rank Parameterization to reduce trainable parameters
    - Confidence-Aware Gating to handle noise sensitivity
    - Entropy Regularization to prevent binary hardness
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device, personalized=False):
        """
        Initializes the NewALAClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task (PyG Data object).
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
            personalized (bool, optional): Flag for personalized FL. Defaults to False.
        """
        super(NewALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized)

        # Get data and masks from the OpenFGL task
        data_obj = self.task.processed_data['data']
        train_mask = self.task.processed_data['train_mask']

        # Create a separate loss function for LoRA-CAAA
        self.lora_caaa_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Initialize the LoRA-CAAA module
        self.LoRA_CAAA = LoRA_CAAA(
            cid=self.client_id,
            loss_fn=self.lora_caaa_loss_fn,
            data_obj=data_obj,
            train_mask=train_mask,
            rand_percent=args.newala_rand_percent,
            layer_idx=args.newala_layer_idx,
            eta=args.newala_eta,
            device=self.device,
            rank=args.newala_rank,
            gamma=args.newala_gamma,
            lambda_reg=args.newala_lambda_reg
        )

    def execute(self):
        """
        Executes the client-side logic for NewALA (LoRA-CAAA):
        1. Receives the global model.
        2. Performs LoRA-CAAA aggregation to initialize the local model.
        3. Trains the local model using the task's standard training method.
        """

        # 1. Receive global model parameters from the server
        received_params = self.message_pool["server"]["weight"]

        # Create a temporary global model object for LoRA-CAAA
        global_model = copy.deepcopy(self.task.model)
        for param, received_param in zip(global_model.parameters(), received_params):
            param.data.copy_(received_param.data)

        # 2. Perform LoRA-CAAA Adaptive Local Aggregation
        self.LoRA_CAAA.adaptive_local_aggregation(global_model, self.task.model)

        # 3. Train the model locally
        self.task.train()

    def send_message(self):
        """
        Sends the locally trained model parameters back to the server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
