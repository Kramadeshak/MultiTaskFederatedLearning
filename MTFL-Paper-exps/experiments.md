### Experiment:1A (FedAvg, Private None, C = 0.5, W = 200, Strategy FL)
Strategy = FL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = None
Algorithm (alg) = fedavg
device = cPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1

**Execution time: ** = 27 mins
**Results pickle file: ** = dset-mnist_alg-fedavg_C-0.5_B-20_T-300_E-1_device-gpu_W-200_seed-0_lr-0.3_noisy_frac-0.0_bn_private-none.pkl

`python main.py -dset mnist -alg fedavg -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private none`

### Experiment:1B (FedAvg-Adam, Private None, C = 0.5, W = 200, Strategy FL)
Strategy = FL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.003
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = None
Algorithm (alg) = fedavg-adam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

**Execution time: ** = 18 mins
**Results pickle file: ** = dset-mnist_alg-fedavg-adam_C-0.5_B-20_T-300_E-1_device-cpu_W-200_seed-0_lr-0.003_noisy_frac-0.0_bn_private-none_beta1-0.9_beta2-0.999_epsilon-1e-07.pkl

`python main.py -dset mnist -alg fedavg-adam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.003 -noisy_frac 0.0 -bn_private none -beta1 0.9 -beta2 0.999 -epsilon 1e-7`

### Experiment:1C (FedAdam, Private None, C = 0.5, W = 200, Strategy FL)
Strategy = FL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = None
Algorithm (alg) = fedadam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-4
server_lr = 0.01

**Execution time: ** = 30 mins
**Results pickle file: ** = dset-mnist_alg-fedadam_C-0.5_B-20_T-300_E-1_device-gpu_W-200_seed-0_lr-0.3_noisy_frac-0.0_bn_private-yb_server_lr-0.01_beta1-0.9_beta2-0.999_epsilon-0.0001.pkl

`python main.py -dset mnist -alg fedadam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private none -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -server_lr 0.01`

### Experiment:2A (FedAvg, Private usyb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = usyb
Algorithm (alg) = fedavg
device = GPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1

**Execution time: ** = 36 mins
**Results pickle file: ** = dset-mnist_alg-fedavg_C-0.5_B-20_T-300_E-1_device-gpu_W-200_seed-0_lr-0.1_noisy_frac-0.0_bn_private-usyb.pkl

`python main.py -dset mnist -alg fedavg -C 0.3 -B 20 -T 300 -E 1 -device gpu -W 200 -seed 0 -lr 0.1 -noisy_frac 0.0 -bn_private usyb`

### Experiment:2B (FedAvg-Adam, Private usyb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.003
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = usyb
Algorithm (alg) = fedavg-adam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

**Execution time: ** = 19 mins
**Results pickle file: ** = dset-mnist_alg-fedavg-adam_C-0.5_B-20_T-300_E-1_device-cpu_W-200_seed-0_lr-0.003_noisy_frac-0.0_bn_private-usyb_beta1-0.9_beta2-0.999_epsilon-1e-07.pkl

`python main.py -dset mnist -alg fedavg-adam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.003 -noisy_frac 0.0 -bn_private usyb -beta1 0.9 -beta2 0.999 -epsilon 1e-7`

### Experiment:2C (FedAdam, Private usyb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = usyb
Algorithm (alg) = fedadam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-4
server_lr = 0.01

**Execution time: ** = 30 mins
**Results pickle file: ** = dset-mnist_alg-fedadam_C-0.5_B-20_T-300_E-1_device-cpu_W-200_seed-0_lr-0.3_noisy_frac-0.0_bn_private-usyb_server_lr-0.01_beta1-0.9_beta2-0.999_epsilon-0.0001.pkl

`python main.py -dset mnist -alg fedadam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private usyb -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -server_lr 0.01`

### Experiment:3A (FedAvg, Private yb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = yb
Algorithm (alg) = fedavg
device = GPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1

**Execution time: ** = 26 mins
**Results pickle file: ** = dset-mnist_alg-fedavg_C-0.5_B-20_T-300_E-1_device-gpu_W-200_seed-0_lr-0.3_noisy_frac-0.0_bn_private-yb.pkl

`python main.py -dset mnist -alg fedavg -C 0.5 -B 20 -T 300 -E 1 -device gpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private yb`

### Experiment:3B (FedAvg-Adam, Private yb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.003
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = yb
Algorithm (alg) = fedavg-adam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

**Execution time: ** = 19 mins
**Results pickle file: ** = dset-mnist_alg-fedavg-adam_C-0.5_B-20_T-300_E-1_device-cpu_W-200_seed-0_lr-0.003_noisy_frac-0.0_bn_private-yb_beta1-0.9_beta2-0.999_epsilon-1e-07.pkl

`python main.py -dset mnist -alg fedavg-adam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.003 -noisy_frac 0.0 -bn_private yb -beta1 0.9 -beta2 0.999 -epsilon 1e-7`

### Experiment:3C (FedAdam, Private yb, C = 0.5, W = 200, Strategy = MTFL)
Strategy = MTFL
Dataset (dset) = MNIST
Clients (W) = 200
Sampled (C) = 0.5
Total communication rounds (T) = 300
Learning rate (lr) = 0.3
Random noise added to fraction of clients (noisy_frac) = 0.0
Private batchnorm parameters (bn_private) = yb
Algorithm (alg) = fedadam
device = CPU
seed = 0
Client batch size (B) = 20
Client num epochs (E) = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-4
server_lr = 0.01

**Execution time: ** = 30 mins
**Results pickle file: ** = dset-mnist_alg-fedadam_C-0.5_B-20_T-300_E-1_device-cpu_W-200_seed-0_lr-0.3_noisy_frac-0.0_bn_private-yb_server_lr-0.01_beta1-0.9_beta2-0.999_epsilon-0.0001.pkl

`python main.py -dset mnist -alg fedadam -C 0.5 -B 20 -T 300 -E 1 -device cpu -W 200 -seed 0 -lr 0.3 -noisy_frac 0.0 -bn_private yb -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -server_lr 0.01`