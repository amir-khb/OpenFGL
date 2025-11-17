import torch
import torch.nn as nn
import numpy as np
import copy
import random
from typing import List, Tuple

from openfgl.flcore.base import BaseClient


# ---
# This ALA class is adapted from your provided ALA.py
# It has been modified to work with PyTorch Geometric (PyG) Data objects
# for GNNs, as used by the OpenFGL framework, instead of a list-based DataLoader.
# The core logic of learning aggregation weights remains the same.
# ---
class ALA:
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
                 num_pre_loss: int = 10) -> None:
        """
        Initialize the GNN-compatible ALA module.

        Args:
            cid: Client ID.
            loss_fn: The loss function (e.g., nn.CrossEntropyLoss).
            data_obj: The PyG Data object for this client.
            train_mask: The train mask for the PyG Data object.
            rand_percent: Percent of local training data to sample for weight learning.
            layer_idx: Control the weight range (from top layers).
            eta: Weight learning rate.
            device: Using cuda or cpu.
            threshold: Convergence threshold for weight learning.
            num_pre_loss: Window size for checking convergence.
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

        self.weights = None  # Learnable local aggregation weights.
        self.start_phase = True

    def adaptive_local_aggregation(self,
                                   global_model: nn.Module,
                                   local_model: nn.Module) -> None:
        """
        Performs adaptive local aggregation by learning weights for parameters.
        This version is adapted for full-batch GNN models.

        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """

        # --- GNN Adaptation ---
        # Instead of a DataLoader, we get a random *mask* of nodes
        # from the training set.
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_indices))
        # --- End GNN Adaptation ---

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                   self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter

        # Ensure data is on the correct device
        self.data_obj = self.data_obj.to(self.device)

        while True:

            # --- GNN Adaptation ---
            # Create a random mask for this weight-learning iteration
            shuffled_indices = self.train_indices[torch.randperm(len(self.train_indices))]
            rand_indices = shuffled_indices[:rand_num]
            rand_mask = torch.zeros_like(self.train_mask, dtype=torch.bool)
            rand_mask[rand_indices] = True
            # --- End GNN Adaptation ---

            optimizer.zero_grad()

            # --- GNN Adaptation ---
            # Full forward pass with the GNN
            # Note: GNN models in OpenFGL typically return (embedding, logits)
            _, output = model_t(self.data_obj)
            # Calculate loss only on the random subset of nodes
            loss_value = self.loss(output[rand_mask], self.data_obj.y[rand_mask])
            # --- End GNN Adaptation ---

            loss_value.backward()

            # update weight in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                       params_gp, self.weights):
                weight.data = torch.clamp(
                    weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

            # update temp local model in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                       params_gp, self.weights):
                param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client: {self.cid}\tStd: {np.std(losses[-self.num_pre_loss:]):.4f}\tALA epochs: {cnt}')
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()


# ---
# This is the FedALAClient class for the OpenFGL framework.
# It inherits from BaseClient and uses the GNN-compatible ALA class.
# ---
class FedALAClient(BaseClient):

    def __init__(self, args, client_id, data, data_dir, message_pool, device, personalized=False):
        """
        Initializes the FedALAClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task (PyG Data object).
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
            personalized (bool, optional): Flag for personalized FL. Defaults to False.
        """
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized)

        # Get data and masks from the OpenFGL task
        # self.task.processed_data contains the client's data and masks
        data_obj = self.task.processed_data['data']
        train_mask = self.task.processed_data['train_mask']

        # Create a separate loss function for ALA, as the task's loss
        # might have different properties (e.g., reduction='none' for DP)
        self.ala_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Initialize the GNN-compatible ALA module
        self.ALA = ALA(
            cid=self.client_id,
            loss_fn=self.ala_loss_fn,
            data_obj=data_obj,
            train_mask=train_mask,
            rand_percent=args.ala_rand_percent,
            layer_idx=args.ala_layer_idx,
            eta=args.ala_eta,
            device=self.device
        )

    def execute(self):
        """
        Executes the client-side logic for FedALA:
        1. Receives the global model.
        2. Performs Adaptive Local Aggregation (ALA) to initialize the local model.
        3. Trains the local model using the task's standard training method.
        """

        # 1. Receive global model parameters from the server
        received_params = self.message_pool["server"]["weight"]

        # Create a temporary global model object for the ALA function
        # We use the task's model as a template
        global_model = copy.deepcopy(self.task.model)
        for param, received_param in zip(global_model.parameters(), received_params):
            param.data.copy_(received_param.data)

        # 2. Perform Adaptive Local Aggregation
        # This function modifies self.task.model in place, blending
        # the local and global models based on learned weights.
        self.ALA.adaptive_local_aggregation(global_model, self.task.model)

        # 3. Train the model locally
        # The task's train() method will now run using the new,
        # ALA-initialized model parameters (self.task.model).
        # This will run for self.args.num_epochs (local steps).
        self.task.train()

    def send_message(self):
        """
        Sends the locally trained model parameters back to the server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,  # Use num_samples from the task
            "weight": list(self.task.model.parameters())
        }