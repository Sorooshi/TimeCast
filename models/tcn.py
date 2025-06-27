"""
Time Series Forecasting Package
Temporal Convolutional Network (TCN) implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from .base_model import BaseTimeSeriesModel

class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()
        # Store parameters as instance variables
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        
        # First dilated causal convolution
        self.conv1 = nn.Conv1d(
            self.n_inputs, self.n_outputs, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        
        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(
            self.n_outputs, self.n_outputs, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(self.n_inputs, self.n_outputs, 1) if self.n_inputs != self.n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights with normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # First convolution block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure residual connection matches the output size
        if out.size(2) != res.size(2):
            # If sizes don't match, trim the longer sequence
            if out.size(2) > res.size(2):
                out = out[:, :, :res.size(2)]
            else:
                res = res[:, :, :out.size(2)]
                
        return self.relu(out + res)

class TCN(BaseTimeSeriesModel):
    def __init__(
        self,
        input_size: int,  # Required parameter, no default
        num_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__()
        # Store all parameters as instance variables
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            # Calculate padding to maintain sequence length
            padding = (self.kernel_size - 1) * dilation
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=self.dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        # Output layer predicts a single value
        self.fc = nn.Linear(self.num_channels[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        # Convert to TCN format: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply TCN
        x = self.network(x)
        
        # Take the last sequence element
        x = x[:, :, -1]
        
        # Project to output size
        x = self.fc(x)
        return x
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            # input_size will be set based on actual data
            'num_channels': [32, 64, 128],
            'kernel_size': 3,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'kernel_size': (2, 5),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-9, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 