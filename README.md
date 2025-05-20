# FL-MTFL: Multi-Task Federated Learning for CIFAR-10

This repository implements Multi-Task Federated Learning (MTFL) with private batch normalization layers as described in the paper [Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing](https://arxiv.org/abs/2007.09236) by Mills et al.

## Features

- Implementation of Federated Learning (FL) using [Flower](https://flower.dev)
- Support for Multi-Task Federated Learning (MTFL) with private batch normalization layers
- Non-IID data partitioning for CIFAR-10
- Support for both SGD and Adam optimizers
- Comprehensive metrics tracking with optional Weights & Biases integration
- Visualization tools for comparing FL and MTFL performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fl-mtfl.git
cd fl-mtfl
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Grant execute permissions:
```bash
chmod +x run_fl.sh
```

4. (Optional) Set up Weights & Biases for experiment tracking:
```bash
pip install wandb
wandb login
```

## Preprocessing

Before running any experiment, preprocess the dataset to create non-IID partitions:

```bash
./run_fl.sh --preprocess
```

## Usage

The main script for running experiments is `run_fl.sh`, which provides a convenient interface for configuring and launching FL and MTFL experiments.

### Experiment Types (`--exp`)

* `fedavg`: Run standard Federated Averaging (FL) experiments.
* `mtfl`: Run Multi-Task Federated Learning experiments with private BatchNorm.
* `both`: Sequentially run both `fedavg` and `mtfl` for comparison.

### Basic Usage

```bash
# Run a simple FedAvg experiment
./run_fl.sh --exp fedavg --num-clients 20 --client-fraction 0.5 --rounds 10

# Run a MTFL experiment with private batch normalization parameters
./run_fl.sh --exp mtfl --bn-private gamma_beta --num-clients 20 --client-fraction 0.5 --rounds 10

# Run both FL and MTFL sequentially for comparison
./run_fl.sh --exp both --bn-private gamma_beta --num-clients 20 --client-fraction 0.5 --rounds 10
```

### Advanced Options

```bash
# Set target accuracy for early stopping
./run_fl.sh --exp mtfl --target-accuracy 0.65 --num-clients 400 --client-fraction 0.5

# Use Adam optimizer instead of SGD
./run_fl.sh --exp mtfl --optimizer adam --learning-rate 0.001 --bn-private gamma_beta

# Save initial weights
./run_fl.sh --save-init --exp fedavg --num-clients 20

# Customize number of local epochs and steps
./run_fl.sh --exp mtfl --local-epochs 1 --steps-per-epoch 10 --per-step-metrics true

# Log metrics to Weights & Biases
./run_fl.sh --exp both --log-wandb --project-name "mtfl-experiments"

# Visualize results after training
./run_fl.sh --exp both --visualize

# Display help
./run_fl.sh --help
```

## Replicating the Paper Results

### Result 1: Communication Rounds to Target Accuracy (Table 2)

```bash
# FL(FedAvg) Baseline
./run_fl.sh --exp fedavg --num-clients 400 --client-fraction 0.5 \
  --target-accuracy 0.65 --max-rounds 500 --optimizer sgd \
  --bn-private none --learning-rate 0.01 \
  --log-wandb --project-name mtfl-replication --experiment-tag "result1-fedavg"

# MTFL(FedAvg) with Private BN (gamma, beta)
./run_fl.sh --exp mtfl --num-clients 400 --client-fraction 0.5 \
  --target-accuracy 0.65 --max-rounds 500 --optimizer sgd \
  --bn-private gamma_beta --learning-rate 0.01 \
  --log-wandb --project-name mtfl-replication --experiment-tag "result1-mtfl-fedavg"

# MTFL(FedAvg-Adam) with Private BN (gamma, beta)
./run_fl.sh --exp mtfl --num-clients 400 --client-fraction 0.5 \
  --target-accuracy 0.65 --max-rounds 500 --optimizer adam \
  --bn-private gamma_beta --learning-rate 0.001 --beta1 0.9 --beta2 0.999 \
  --log-wandb --project-name mtfl-replication --experiment-tag "result1-mtfl-fedavg-adam"
```

### Result 2: Train/Test UA Curves (Figure 4)

```bash
./run_fl.sh --exp mtfl --num-clients 200 --client-fraction 1.0 \
  --local-epochs 1 --per-step-metrics true --steps-per-epoch 10 \
  --bn-private gamma_beta --optimizer sgd --learning-rate 0.01 \
  --log-wandb --project-name mtfl-replication --experiment-tag "result2-mtfl-gamma-beta"
```

### Result 3: Per-Round Test UA Comparison (Figure 5)

```bash
./run_fl.sh --exp mtfl --num-clients 400 --client-fraction 0.5 \
  --max-rounds 200 --optimizer sgd --bn-private gamma_beta --learning-rate 0.01 \
  --seed 42 --visualize \
  --log-wandb --project-name mtfl-replication --experiment-tag "result3-mtfl-fedavg-seed1"
```

To run with multiple seeds:

```bash
for SEED in 42 43 44 45 46; do
  ./run_fl.sh --exp fedavg --num-clients 400 --client-fraction 0.5 \
    --max-rounds 200 --optimizer sgd --bn-private none --learning-rate 0.01 \
    --seed $SEED --visualize \
    --log-wandb --project-name mtfl-replication --experiment-tag "result3-fedavg-seed$SEED" &
done
wait
```

Then combine metrics:

```bash
python visualize_results.py --metrics-dir "./metrics" --figures-dir "./figures" \
  --experiment-tag "result3" --confidence-interval 0.95 --seeds 5
```

## Outputs

* Training logs and metrics: `./metrics/`
* Visualization plots: `./figures/`
* Initial model weights: `initial_weights.pth`

## Key Components

* `model.py`: CNN model for CIFAR-10 with optional private batch normalization
* `task.py`: Core functions for training, evaluation, and data handling
* `client_app.py`: Flower client implementation
* `server_app.py`: Flower server implementation with metric tracking
* `config.py`: Configuration system for experiments
* `wandb_utils.py`: Utilities for Weights & Biases integration
* `preprocess_data.py`: Script for creating non-IID partitions
* `visualize_results.py`: Script for creating result visualizations
* `run_fl.sh`: Main script to run experiments