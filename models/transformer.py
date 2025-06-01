import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple
from .base_model import BaseTimeSeriesModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Store parameters as instance variables
        self.d_model = d_model  # dimensionality of the model's internal representations
        self.max_len = max_len  # maximum sequence length that the positional encoding can handle.
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # computing the frequency terms for the sinusoidal position encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * # Creates a sequence from 0 to d_model-1, stepping by 2 (Positional encoding)
            (-math.log(10000.0) / d_model) # controls how quickly the frequencies change
        )  # creates different frequencies for the positional encoding        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class Transformer(BaseTimeSeriesModel):
    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        # Store all parameters as instance variables
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Input projection: converts the input features to the model's working dimension
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding: positional information to the input sequence because the 
        # Transformer itself has no built-in way to understand the order of elements in a sequence.
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection now predicts a single value (total consumption)
        self.output_projection = nn.Linear(self.d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create target mask for the last position
        target_seq = x[:, -1:, :]
        
        # Generate prediction
        output = self.transformer(
            src=x,
            tgt=target_seq,
            src_mask=None,
            tgt_mask=None,
            memory_mask=None
        )
        
        # Project output back to a single value (total consumption)
        prediction = self.output_projection(output[:, -1, :])
        return prediction
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            'input_size': 10,  # This will be overridden by actual data
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'learning_rate': 1e-4
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'd_model': (16, 256),
            'nhead': (2, 8),
            'num_encoder_layers': (1, 6),
            'num_decoder_layers': (1, 6),
            'dim_feedforward': (32, 512),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-6, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 