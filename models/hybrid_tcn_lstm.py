"""
Time Series Forecasting Package
Hybrid TCN-LSTM model implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from .base_model import BaseTimeSeriesModel
import numpy as np

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
            stride=self.stride, padding=0, dilation=self.dilation
        )
        self.chomp1 = nn.Identity()  # Remove time dimension chomping
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        
        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(
            self.n_outputs, self.n_outputs, self.kernel_size,
            stride=self.stride, padding=0, dilation=self.dilation
        )
        self.chomp2 = nn.Identity()  # Remove time dimension chomping
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
        # Calculate padding to maintain sequence length
        padding = (self.kernel_size - 1) * self.dilation
        x_pad = nn.functional.pad(x, (padding, 0))
        
        # First convolution block
        out = self.conv1(x_pad)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution block with padding
        out_pad = nn.functional.pad(out, (padding, 0))
        out = self.conv2(out_pad)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class HybridTCNLSTM(BaseTimeSeriesModel):
    def __init__(
        self,
        input_size: int,  # Required parameter, no default
        tcn_channels: List[int] = [32, 64],  # Reduced complexity in TCN part
        kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        # Store all parameters as instance variables
        self.input_size = input_size
        self.tcn_channels = tcn_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Build TCN layers
        tcn_layers = []
        num_levels = len(self.tcn_channels)
        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = self.input_size if i == 0 else self.tcn_channels[i-1]
            out_channels = self.tcn_channels[i]
            
            tcn_layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=0,  # We'll handle padding in the forward pass
                    dropout=self.dropout
                )
            )
        
        self.tcn = nn.Sequential(*tcn_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.tcn_channels[-1],  # Input size is last TCN channel size
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.dropout if self.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Final output layer
        self.fc = nn.Linear(self.lstm_hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        # TCN expects input shape (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        tcn_out = self.tcn(x)
        
        # Prepare for LSTM (batch_size, sequence_length, channels)
        lstm_in = tcn_out.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Use last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        prediction = self.fc(last_output)
        return prediction
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            # input_size will be set based on actual data
            'tcn_channels': [32, 64],
            'kernel_size': 3,
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 1e-5
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'kernel_size': (2, 5),
            'lstm_hidden_size': (32, 256),
            'lstm_num_layers': (1, 4),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-9, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 