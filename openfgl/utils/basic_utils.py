import torch
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp

import os
import pickle
import random
import sys  # <-- ADD THIS IMPORT


# --- ADD THIS FUNCTION ---
def total_size(o):
    """
    Recursively calculates the total size in bytes of a Python object.
    Handles tensors, dictionaries, lists, and other basic types.
    """
    if isinstance(o, torch.Tensor):
        # Return size of tensor data
        return o.element_size() * o.nelement()

    if isinstance(o, dict):
        # Return size of dict plus size of its keys and values
        return sys.getsizeof(o) + sum(total_size(k) + total_size(v) for k, v in o.items())

    if isinstance(o, (list, tuple)):
        # Return size of list/tuple plus size of its items
        return sys.getsizeof(o) + sum(total_size(item) for item in o)

    if isinstance(o, (int, float, str, bool)) or o is None:
        # Return size of basic types
        return sys.getsizeof(o)

    # For other types, return the basic size
    return sys.getsizeof(o)


# --- END OF FUNCTION ---

def load_task(args, client_id, data, data_dir, device):
    if args.task == "node_cls":
        from openfgl.task.node_cls import NodeClsTask
        task = NodeClsTask(args, client_id, data, data_dir, device)
    elif args.task == "link_pred":
        from openfgl.task.link_pred import LinkPredTask
        task = LinkPredTask(args, client_id, data, data_dir, device)
    elif args.task == "graph_cls":
        from openfgl.task.graph_cls import GraphClsTask
        task = GraphClsTask(args, client_id, data, data_dir, device)
    elif args.task == "node_clust":
        from openfgl.task.node_clust import NodeClustTask
        task = NodeClustTask(args, client_id, data, data_dir, device)
    else:
        raise NotImplementedError(f"task {args.task} has not been implemented.")
    return task


def load_client(args, client_id, data, data_dir, message_pool, device):
    fl_algorithm = args.fl_algorithm
    if fl_algorithm == "isolate":
        from openfgl.flcore.isolate.client import IsolateClient
        return IsolateClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedavg":
        from openfgl.flcore.fedavg.client import FedAvgClient
        return FedAvgClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedprox":
        from openfgl.flcore.fedprox.client import FedProxClient
        return FedProxClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "scaffold":
        from openfgl.flcore.scaffold.client import ScaffoldClient
        return ScaffoldClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "moon":
        from openfgl.flcore.moon.client import MoonClient
        return MoonClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "feddc":
        from openfgl.flcore.feddc.client import FedDCClient
        return FedDCClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedproto":
        from openfgl.flcore.fedproto.client import FedProtoClient
        return FedProtoClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedtgp":
        from openfgl.flcore.fedtgp.client import FedTGPClient
        return FedTGPClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedpub":
        from openfgl.flcore.fedpub.client import FedPubClient
        return FedPubClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedstar":
        from openfgl.flcore.fedstar.client import FedStarClient
        return FedStarClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedgta":
        from openfgl.flcore.fedgta.client import FedGTAClient
        return FedGTAClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedtad":
        from openfgl.flcore.fedtad.client import FedTADClient
        return FedTADClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "gcfl_plus":
        from openfgl.flcore.gcfl_plus.client import GCFLPlusClient
        return GCFLPlusClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedsage_plus":
        from openfgl.flcore.fedsage_plus.client import FedSagePlusClient
        return FedSagePlusClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "adafgl":
        from openfgl.flcore.adafgl.client import AdaFGLClient
        return AdaFGLClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "feddep":
        from openfgl.flcore.feddep.client import FedDEPClient
        return FedDEPClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fggp":
        from openfgl.flcore.fggp.client import FGGPClient
        return FGGPClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fgssl":
        from openfgl.flcore.fgssl.client import FGSSLClient
        return FGSSLClient(args, client_id, data, data_dir, message_pool, device)
    elif fl_algorithm == "fedgl":
        from openfgl.flcore.fedgl.client import FedGLClient
        return FedGLClient(args, client_id, data, data_dir, message_pool, device, personalized=True)
    # --- ADD THIS BLOCK FOR FEDALA ---
    elif fl_algorithm == "fedala":
        from openfgl.flcore.fedala.client import FedALAClient
        return FedALAClient(args, client_id, data, data_dir, message_pool, device)
    # --- END OF BLOCK ---


def load_server(args, global_data, data_dir, message_pool, device):
    fl_algorithm = args.fl_algorithm
    if fl_algorithm == "isolate":
        from openfgl.flcore.isolate.server import IsolateServer
        return IsolateServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedavg":
        from openfgl.flcore.fedavg.server import FedAvgServer
        return FedAvgServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedprox":
        from openfgl.flcore.fedprox.server import FedProxServer
        return FedProxServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "scaffold":
        from openfgl.flcore.scaffold.server import ScaffoldServer
        return ScaffoldServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "moon":
        from openfgl.flcore.moon.server import MoonServer
        return MoonServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "feddc":
        from openfgl.flcore.feddc.server import FedDCServer
        return FedDCServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedproto":
        from openfgl.flcore.fedproto.server import FedProtoServer
        return FedProtoServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedtgp":
        from openfgl.flcore.fedtgp.server import FedTGPServer
        return FedTGPServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedpub":
        from openfgl.flcore.fedpub.server import FedPubServer
        return FedPubServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedstar":
        from openfgl.flcore.fedstar.server import FedStarServer
        return FedStarServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedgta":
        from openfgl.flcore.fedgta.server import FedGTAServer
        return FedGTAServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedtad":
        from openfgl.flcore.fedtad.server import FedTADServer
        return FedTADServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "gcfl_plus":
        from openfgl.flcore.gcfl_plus.server import GCFLPlusServer
        return GCFLPlusServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fedsage_plus":
        from openfgl.flcore.fedsage_plus.server import FedSagePlusServer
        return FedSagePlusServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "adafgl":
        from openfgl.flcore.adafgl.server import AdaFGLServer
        return AdaFGLServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "feddep":
        from openfgl.flcore.feddep.server import FedDEPServer
        return FedDEPServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fggp":
        from openfgl.flcore.fggp.server import FGGPServer
        return FGGPServer(args, global_data, data_dir, message_pool, device, personalized=True)
    elif fl_algorithm == "fgssl":
        from openfgl.flcore.fgssl.server import FGSSLServer
        return FGSSLServer(args, global_data, data_dir, message_pool, device)
    elif fl_algorithm == "fedgl":
        from openfgl.flcore.fedgl.server import FedGLServer
        return FedGLServer(args, global_data, data_dir, message_pool, device, personalized=True)
    # --- ADD THIS BLOCK FOR FEDALA ---
    elif fl_algorithm == "fedala":
        from openfgl.flcore.fedala.server import FedALAServer
        return FedALAServer(args, global_data, data_dir, message_pool, device)
    # --- END OF BLOCK ---


def save_checkpoints(args, client_id, model, optim, round_):
    path = f"./checkpoints/{args.scenario}/{args.simulation_mode}_{args.louvain_resolution}_{args.dataset[0]}_client_{args.num_clients}/{args.fl_algorithm}_{args.task}_{args.model[0]}_seed_{args.seed}"
    if not os.path.exists(path):
        os.makedirs(path)
    path = f"{path}/client_{client_id}_round_{round_}.pt"
    torch.save({
        'round': round_,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, path)


def load_checkpoints(args, client_id, model, optim, round_):
    path = f"./checkpoints/{args.scenario}/{args.simulation_mode}_{args.louvain_resolution}_{args.dataset[0]}_client_{args.num_clients}/{args.fl_algorithm}_{args.task}_{args.model[0]}_seed_{args.seed}"
    path = f"{path}/client_{client_id}_round_{round_}.pt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_round = checkpoint['round']
    return model, optim, start_round


def write_log(args, log):
    path = f"./logs/{args.scenario}/{args.simulation_mode}_{args.louvain_resolution}_{args.dataset[0]}_client_{args.num_clients}/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = f"{path}/{args.fl_algorithm}_{args.task}_{args.model[0]}_seed_{args.seed}.log"

    with open(path, "a") as f:
        f.write(log)
        f.write("\n")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def add_connections(data, num_con):
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    # avoid self-loop
    adj.setdiag(0)
    adj.eliminate_zeros()

    # find all zero entries
    adj_ = adj.tocoo()

    row, col = adj_.row, adj_.col
    v_set = set(zip(row, col))

    all_v = set(zip(range(data.num_nodes), range(data.num_nodes)))

    for i in range(data.num_nodes):
        for j in range(i + 1, data.num_nodes):
            all_v.add((i, j))
            all_v.add((j, i))

    false_v_set = all_v - v_set

    # add connections
    if len(false_v_set) < num_con:
        # print(f"can only add {len(false_v_set)} edges")
        num_con = len(false_v_set)

    if num_con > 0:
        new_v_list = random.sample(false_v_set, num_con)

        new_row = [v[0] for v in new_v_list]
        new_col = [v[1] for v in new_v_list]

        adj.row = np.hstack((adj.row, new_row))
        adj.col = np.hstack((adj.col, new_col))

    edge_index, _ = from_scipy_sparse_matrix(adj)
    return edge_index


def extract_floats(s):
    return [float(x) for x in s.split('-')]


def idx_to_mask_tensor(idx, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[idx] = 1
    return mask


def mask_tensor_to_idx(mask):
    return mask.nonzero(as_tuple=False).squeeze(-1).tolist()