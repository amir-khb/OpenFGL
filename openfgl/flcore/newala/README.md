# NewALA: LoRA-CAAA Framework

## Overview

NewALA implements the **LoRA-CAAA** (Low-Rank Confidence-Aware Adaptive Aggregation) framework, which is an enhanced version of the Federated Adaptive Local Aggregation (FedALA) algorithm. This method addresses three key challenges in federated learning on graphs:

1. **Parameter Explosion** - Solved by Low-Rank Parameterization
2. **Noise Sensitivity** - Solved by Confidence-Aware Gating
3. **Binary Hardness** - Solved by Entropy Regularization

## Key Components

### 1. Low-Rank Parameterization (Solving Parameter Explosion)

Instead of learning the full-rank tensor W ∈ ℝ^(m×n), we decompose the aggregation weights into low-rank matrices:

- A ∈ ℝ^(n×r)
- B ∈ ℝ^(m×r)

where r ≪ min(m,n)

**Formula:**
```
W = σ(A · B^T)
```

This reduces trainable parameters from m × n to (m + n) × r.

### 2. Confidence-Aware Gating (Solving Noise Sensitivity)

We introduce a dynamic scalar β that scales the update based on the model's confidence in the current data batch.

**Batch Entropy:**
```
E(x) = -1/N Σ Σ p_k(x_i) log p_k(x_i)
```

**Trust Coefficient:**
```
β = exp(-γ · E(x))
```

When the model is uncertain (high entropy), β → 0, suppressing the aggregation update.

### 3. Entropy Regularization (Solving Binary Hardness)

To prevent W from collapsing to binary values, we add a regularization term:

```
R(W) = -Σ [W_ij log W_ij + (1 - W_ij) log(1 - W_ij)]
```

### 4. Unified Algorithm

Combining all three components:

**Initialization:**
```
θ_init = θ_l + β · [(θ_g - θ_l) ⊙ σ(A · B^T)]
```

**Optimization:**
```
min_{A,B} (L_task(θ_init) - λ R(σ(A · B^T)))
```

where:
- θ_l: local model parameters
- θ_g: global model parameters
- β: trust coefficient (confidence-aware gating)
- A, B: low-rank matrices
- λ: regularization weight

## Usage

### Command Line

```bash
python test_newala.py
```

Or with custom parameters:

```bash
python main.py \
    --fl_algorithm newala \
    --dataset Cora \
    --model gcn \
    --num_clients 10 \
    --num_rounds 100 \
    --num_epochs 3 \
    --newala_rank 4 \
    --newala_gamma 0.1 \
    --newala_lambda_reg 0.01 \
    --newala_eta 1.0 \
    --newala_rand_percent 80 \
    --newala_layer_idx 2
```

### Python API

```python
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

args = config.args
args.fl_algorithm = "newala"
args.dataset = ["Cora"]
args.model = ["gcn"]

# NewALA-specific parameters
args.newala_rank = 4
args.newala_gamma = 0.1
args.newala_lambda_reg = 0.01

trainer = FGLTrainer(args)
trainer.train()
```

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `newala_rank` | int | 4 | Rank r for low-rank decomposition (r ≪ min(m,n)) |
| `newala_gamma` | float | 0.1 | Sensitivity hyperparameter γ for confidence-aware gating |
| `newala_lambda_reg` | float | 0.01 | Regularization weight λ for entropy regularization |
| `newala_eta` | float | 1.0 | Learning rate η for low-rank matrix optimization |
| `newala_rand_percent` | int | 80 | Percentage of training data to sample for weight learning |
| `newala_layer_idx` | int | 2 | Number of layers (from top) to apply NewALA |

## Comparison with FedALA

| Aspect | FedALA | NewALA (LoRA-CAAA) |
|--------|--------|---------------------|
| Parameter Count | m × n | (m + n) × r |
| Noise Handling | Fixed weighting | Dynamic confidence-based weighting |
| Weight Regularization | None | Entropy regularization |
| Convergence | May collapse to binary | Prevented by regularization |

## Key Advantages

1. **Reduced Memory**: Low-rank decomposition reduces memory usage significantly
2. **Robustness**: Confidence-aware gating handles noisy/unreliable data better
3. **Stability**: Entropy regularization prevents weight collapse
4. **Flexibility**: Can adjust rank r based on available resources

## References

Based on the theoretical framework presented in Section 3 of the paper:
- Section 3.1: Low-Rank Parameterization
- Section 3.2: Confidence-Aware Gating
- Section 3.3: Entropy Regularization
- Section 3.4: The Proposed Unified Algorithm

## Files

- `client.py`: Implements the LoRA-CAAA client logic
- `server.py`: Implements the FedAvg server-side aggregation
- `__init__.py`: Module initialization

## Implementation Details

### Client-Side (`client.py`)

The `LoRA_CAAA` class implements:
- `low_rank_decomposition()`: Initializes A and B matrices using SVD
- `compute_entropy()`: Calculates batch entropy E(x)
- `compute_trust_coefficient()`: Computes β = exp(-γ·E(x))
- `compute_entropy_regularization()`: Computes R(W)
- `adaptive_local_aggregation()`: Main aggregation algorithm

### Server-Side (`server.py`)

The server uses standard FedAvg aggregation (weighted averaging).

## Notes

- The implementation is optimized for PyTorch Geometric (PyG) data objects
- Works with GNN models (GCN, GraphSAGE, GAT, etc.)
- Compatible with the OpenFGL framework
- Supports GPU acceleration
