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
        
        # Create position matrix [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # computing the frequency terms for the sinusoidal position encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # Calculate number of dimensions to use for sine/cosine
        n_dims = d_model // 2 * 2  # Round down to nearest even number
        
        # Compute positional encoding for even indices
        pe_even = torch.sin(position * div_term[:n_dims//2])
        pe_odd = torch.cos(position * div_term[:n_dims//2])
        
        # Interleave the sine and cosine values
        pe[:, 0:n_dims:2] = pe_even
        pe[:, 1:n_dims:2] = pe_odd
        
        # If d_model is odd, handle the last dimension
        if d_model % 2 == 1:
            last_dim = torch.sin(position * div_term[-1])
            pe[:, -1] = last_dim.squeeze(-1)
        
        # Add batch dimension and register buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor."""
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
        # Validate parameters
        # Ensure nhead is even
        if nhead % 2 != 0:
            nhead = nhead + 1
            print(f"Warning: nhead must be even. Adjusted nhead to {nhead}")
            
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            # Round d_model up to the nearest multiple of nhead
            d_model = ((d_model + nhead - 1) // nhead) * nhead
            print(f"Warning: d_model must be divisible by nhead. Adjusted d_model to {d_model}")
        
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
        
        # Output projection predicts a single value
        self.output_projection = nn.Linear(self.d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create target sequence (using last position)
        target_seq = x[:, -1:, :]
        
        # Generate prediction
        output = self.transformer(
            src=x,
            tgt=target_seq,
            src_mask=None,
            tgt_mask=None,
            memory_mask=None
        )
        
        # Project output to prediction
        prediction = self.output_projection(output[:, -1, :])
        return prediction
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            'input_size': 10,  # This will be overridden by actual data
            'd_model': 64,     # Must be divisible by nhead
            'nhead': 4,        # Must be even for optimal performance
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'learning_rate': 1e-4
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'd_model': (32, 256),  # Will be adjusted to be divisible by nhead
            'nhead': (2, 8),       # Will be adjusted to be even
            'num_encoder_layers': (1, 6),
            'num_decoder_layers': (1, 6),
            'dim_feedforward': (32, 512),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-6, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 