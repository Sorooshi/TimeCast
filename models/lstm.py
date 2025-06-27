"""
Time Series Forecasting Package
LSTM model implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from models.base_model import BaseTimeSeriesModel

class LSTM(BaseTimeSeriesModel):
    """LSTM model for time series forecasting"""
    
    def __init__(
        self,
        input_size: int,  # Required parameter, no default
        hidden_size: int = 64,
        num_layers: int = 2, 
        dropout: float = 0.1,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout when num_layers > 1
            batch_first=True
        )
        
        # Output layer now predicts a single value
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the last output for prediction
        return self.fc(lstm_out[:, -1, :])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Return default parameters for the model"""
        return {
            # input_size will be set based on actual data
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001
        }
        
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges for hyperparameter tuning"""
        return {
            'hidden_size': (8, 256),  # Will be converted to int by Optuna
            'num_layers': (1, 6),      # Will be converted to int by Optuna
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-9, 1e-3)
        } 