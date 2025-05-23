#activate the virtual environment
#source ../../../PyTorch/py_torch_124env/Scripts/activate

import torchvision
import os

# Create directories if they don't exist
mnist_path = "../MNIST_data"
cifar_path = "../CIFAR10_data"

os.makedirs(mnist_path, exist_ok=True)
os.makedirs(cifar_path, exist_ok=True)

# Download MNIST dataset
print("Downloading MNIST dataset...")
torchvision.datasets.MNIST(root=mnist_path, train=True, download=True)
torchvision.datasets.MNIST(root=mnist_path, train=False, download=True)

# Download CIFAR-10 dataset
print("Downloading CIFAR-10 dataset...")
torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=True)
torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=True)