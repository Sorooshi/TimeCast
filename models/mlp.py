"""
TimeCast
Hybrid TCN-LSTM model implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_model import BaseTimeSeriesModel

class MLP(BaseTimeSeriesModel):
    """Simple Multi-Layer Perceptron for time series forecasting"""
    
    def __init__(
        self,
        input_size: int,  # Required parameter, no default
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        sequence_length: int = 10  # Add sequence_length parameter
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        
        # Build MLP layers
        layers = []
        
        # First layer takes flattened input (sequence_length * input_size features)
        prev_size = sequence_length * input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        batch_size = x.size(0)
        # Flatten the input: (batch_size, sequence_length * input_size)
        x = x.reshape(batch_size, -1)
        
        # Pass through MLP
        return self.network(x)
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Return default parameters for the model"""
        return {
            # input_size will be set based on actual data
            'hidden_sizes': [64, 32],
            'dropout': 0.1,
            'learning_rate': 0.001,
            'sequence_length': 10  # Add default sequence_length
        }
        
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges for hyperparameter tuning"""
        return {
            'hidden_sizes': [(8, 64), (8, 128), (16, 256), (64, 256)],  # Will be handled separately in training.py
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-9, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 