"""
Model definition for CIFAR-10 classification.

This module defines the CNN model architecture for CIFAR-10,
with support for private batch normalization layers as required
by the Multi-Task Federated Learning (MTFL) approach.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.model")


class CIFAR10CNN(nn.Module):
    """CNN model for CIFAR-10 classification with optional private batch normalization.
    
    This model can be configured to use private batch normalization layers,
    which means the batch normalization parameters (gamma, beta) will not be
    shared during federated averaging, as described in the MTFL approach.
    """
    
    def __init__(self, bn_private: str = "none"):
        """Initialize model architecture.
        
        Parameters
        ----------
        bn_private : str, optional
            Which BN parameters to keep private:
            - "none": No private parameters (regular FedAvg)
            - "gamma_beta": Keep gamma and beta parameters private
            - "mu_sigma": Keep running mean and variance private
            - "all": Keep all BN parameters private
            Default is "none".
        """
        super().__init__()
        self.bn_private = bn_private
        self.use_private_bn = bn_private != "none"
        
        # Store which parameters are private
        self.private_gamma_beta = bn_private in ["gamma_beta", "all"]
        self.private_mu_sigma = bn_private in ["mu_sigma", "all"]
        
        # Log the model configuration
        logger.info(f"Initializing CIFAR10CNN with bn_private={bn_private}")
        
        # Define main convolutional and fully connected layers
        # These will be shared during federated averaging
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Define batch normalization layers with appropriate settings
        affine = not self.private_gamma_beta
        track_running_stats = not self.private_mu_sigma
        
        self.bn1 = nn.BatchNorm2d(32, affine=affine, track_running_stats=track_running_stats)
        self.bn2 = nn.BatchNorm2d(64, affine=affine, track_running_stats=track_running_stats)
        self.bn3 = nn.BatchNorm1d(512, affine=affine, track_running_stats=track_running_stats)
        
        # If using private gamma/beta, add separate affine transformations
        if self.private_gamma_beta:
            logger.info("Adding private affine transformations for batch normalization")
            self.bn1_affine = nn.Parameter(torch.ones(32))
            self.bn1_bias = nn.Parameter(torch.zeros(32))
            
            self.bn2_affine = nn.Parameter(torch.ones(64))
            self.bn2_bias = nn.Parameter(torch.zeros(64))
            
            self.bn3_affine = nn.Parameter(torch.ones(512))
            self.bn3_bias = nn.Parameter(torch.zeros(512))
        
        # If using private mu/sigma, add buffers for running statistics
        if self.private_mu_sigma:
            logger.info("Adding private running statistics for batch normalization")
            self.register_buffer('bn1_mean', torch.zeros(32))
            self.register_buffer('bn1_var', torch.ones(32))
            
            self.register_buffer('bn2_mean', torch.zeros(64))
            self.register_buffer('bn2_var', torch.ones(64))
            
            self.register_buffer('bn3_mean', torch.zeros(512))
            self.register_buffer('bn3_var', torch.ones(512))

    def _apply_bn_with_private_affine(self, x: torch.Tensor, bn: nn.BatchNorm2d, affine: nn.Parameter, bias: nn.Parameter) -> torch.Tensor:
        """Apply batch normalization with private affine transformation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        bn : nn.BatchNorm2d or nn.BatchNorm1d
            Batch normalization layer.
        affine : nn.Parameter
            Scale parameter for affine transformation.
        bias : nn.Parameter
            Bias parameter for affine transformation.
            
        Returns
        -------
        torch.Tensor
            Normalized and transformed tensor.
        """
        # Apply batch normalization without affine transformation
        x = bn(x)
        
        # Apply private affine transformation
        if x.dim() == 4:  # For conv layers (B, C, H, W)
            return x * affine.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        else:  # For fully connected layers (B, C)
            return x * affine.view(1, -1) + bias.view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input batch of images, shape [B, 3, 32, 32]
            
        Returns
        -------
        torch.Tensor
            Output logits, shape [B, 10]
        """
        # First convolutional block
        x = self.conv1(x)
        if self.private_gamma_beta:
            x = self._apply_bn_with_private_affine(x, self.bn1, self.bn1_affine, self.bn1_bias)
        else:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second convolutional block
        x = self.conv2(x)
        if self.private_gamma_beta:
            x = self._apply_bn_with_private_affine(x, self.bn2, self.bn2_affine, self.bn2_bias)
        else:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        if self.private_gamma_beta:
            x = self._apply_bn_with_private_affine(x, self.bn3, self.bn3_affine, self.bn3_bias)
        else:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

    def get_shared_parameters(self) -> List[nn.Parameter]:
        """Get list of global parameters that should be shared during federated averaging.
        
        Returns
        -------
        List[nn.Parameter]
            List of global parameters.
        """
        global_params = []
        
        # Add all parameters from layers that are always shared
        global_params.extend(self.conv1.parameters())
        global_params.extend(self.conv2.parameters())
        global_params.extend(self.fc1.parameters())
        global_params.extend(self.fc2.parameters())
        
        # Add batch normalization parameters if not using private BN
        if not self.private_gamma_beta:
            # Add gamma/beta parameters if not private
            for bn in [self.bn1, self.bn2, self.bn3]:
                for name, param in bn.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        global_params.append(param)
        
        return global_params

    def get_private_parameters(self) -> List[nn.Parameter]:
        """Get list of private parameters that should not be shared during federated averaging.
        
        Returns
        -------
        List[nn.Parameter]
            List of private parameters.
        """
        if not self.use_private_bn:
            return []
        
        private_params = []
        
        # Add private gamma/beta if specified
        if self.private_gamma_beta and hasattr(self, 'bn1_affine'):
            private_params.extend([
                self.bn1_affine, self.bn1_bias,
                self.bn2_affine, self.bn2_bias,
                self.bn3_affine, self.bn3_bias
            ])
        
        return private_params
    
    def get_bn_private_buffers(self) -> Dict[str, torch.Tensor]:
        """Get private BN running statistics as a dictionary.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of private BN running statistics.
        """
        if not self.private_mu_sigma:
            return {}
        
        return {
            'bn1_mean': self.bn1_mean.clone(),
            'bn1_var': self.bn1_var.clone(),
            'bn2_mean': self.bn2_mean.clone(),
            'bn2_var': self.bn2_var.clone(),
            'bn3_mean': self.bn3_mean.clone(),
            'bn3_var': self.bn3_var.clone()
        }
    
    def set_bn_private_buffers(self, buffers: Dict[str, torch.Tensor]) -> None:
        """Set private BN running statistics from a dictionary.
        
        Parameters
        ----------
        buffers : Dict[str, torch.Tensor]
            Dictionary of private BN running statistics.
        """
        if not self.private_mu_sigma or not buffers:
            return
        
        if 'bn1_mean' in buffers:
            self.bn1_mean.copy_(buffers['bn1_mean'])
        if 'bn1_var' in buffers:
            self.bn1_var.copy_(buffers['bn1_var'])
        if 'bn2_mean' in buffers:
            self.bn2_mean.copy_(buffers['bn2_mean'])
        if 'bn2_var' in buffers:
            self.bn2_var.copy_(buffers['bn2_var'])
        if 'bn3_mean' in buffers:
            self.bn3_mean.copy_(buffers['bn3_mean'])
        if 'bn3_var' in buffers:
            self.bn3_var.copy_(buffers['bn3_var'])