import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_model import BaseTimeSeriesModel

class LSTM(BaseTimeSeriesModel):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        # Store all parameters as instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer now predicts a single value (total consumption)
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Predict total consumption
        prediction = self.fc(last_output)
        return prediction
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'hidden_size': (16, 256),
            'num_layers': (1, 10),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-6, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 