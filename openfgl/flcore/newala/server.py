import torch
from openfgl.flcore.base import BaseServer


class NewALAServer(BaseServer):
    """
    NewALAServer implements the server-side logic for the NewALA (LoRA-CAAA) algorithm.

    The NewALA algorithm's unique logic (LoRA-CAAA: Low-Rank Confidence-Aware
    Adaptive Aggregation) is performed on the client side. The server-side
    aggregation is identical to Federated Averaging (FedAvg).

    This class is responsible for aggregating model updates from clients
    and broadcasting the updated global model.

    Attributes:
        (inherits attributes from BaseServer)
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the NewALAServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(NewALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        """
        Executes the server-side aggregation (FedAvg).

        This method aggregates model updates from the clients by computing a
        weighted average of the model parameters, based on the number
        of samples each client used for training.
        """
        with torch.no_grad():
            # Sum the number of samples from all sampled clients
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])

            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                # Get client's weight (num_samples / total_samples)
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples

                # Get client's model parameters
                client_params = self.message_pool[f"client_{client_id}"]["weight"]

                # Perform weighted aggregation
                for (local_param, global_param) in zip(client_params, self.task.model.parameters()):
                    if it == 0:
                        # For the first client, copy the weighted parameters
                        global_param.data.copy_(weight * local_param)
                    else:
                        # For subsequent clients, add the weighted parameters
                        global_param.data += weight * local_param

    def send_message(self):
        """
        Sends a message to the clients containing the updated global model
        parameters after aggregation.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }
