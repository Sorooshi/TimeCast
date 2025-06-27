"""
Time Series Forecasting Package
Hybrid TCN-LSTM model implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple, Optional
from .base_model import BaseTimeSeriesModel


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        patch_len: int,
        d_model: int
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_len = patch_len
        self.d_model = d_model
        
        # Linear projection for each patch
        self.projection = nn.Linear(patch_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input time series into patch embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, input_size, d_model)
        """
        # Reshape into patches
        batch_size, seq_len, _ = x.size()
        num_patches = seq_len // self.patch_len # + 2
        
        # Ensure sequence length is divisible by patch_len
        if seq_len % self.patch_len != 0:
            pad_len = self.patch_len - (seq_len % self.patch_len)
            x = torch.pad(x, (0, 0, 0, pad_len))
            num_patches = (seq_len + pad_len) // self.patch_len
            
        # Reshape into patches: (batch, num_patches, patch_len, input_size)
        x = x.reshape(batch_size, num_patches, self.patch_len, self.input_size)
        
        # Transpose to get (batch, num_patches, input_size, patch_len)
        x = x.transpose(-1, -2)
        
        # Project each patch: (batch, num_patches, input_size, d_model)
        return self.projection(x)

class ChannelIndependentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Separate Q, K, V projections for each head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply channel-independent multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, num_patches, input_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Attention output of same shape as input
        """
        batch_size, num_patches, input_size, _ = x.size()
        
        # Project queries, keys, and values
        q = self.q_proj(x)  # (batch, num_patches, input_size, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention computation
        scale = math.sqrt(self.d_model // self.n_heads)
        q = q.reshape(batch_size, num_patches, input_size, self.n_heads, -1).transpose(2, 3)
        k = k.reshape(batch_size, num_patches, input_size, self.n_heads, -1).transpose(2, 3)
        v = v.reshape(batch_size, num_patches, input_size, self.n_heads, -1).transpose(2, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(2, 3).reshape(batch_size, num_patches, input_size, self.d_model)
        return self.out_proj(out)

class PatchTST(BaseTimeSeriesModel):
    def __init__(
        self,
        input_size: int,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        # Store parameters
        self.input_size = input_size
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(input_size, patch_len, d_model)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1000, 1, d_model)  # Max 1000 patches supported
        )
        
        # Transformer layers with channel-independent attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': ChannelIndependentAttention(d_model, n_heads, dropout),
                'norm1': nn.LayerNorm([input_size, d_model]),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm([input_size, d_model])
            }) for _ in range(n_layers)
        ])
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PatchTST.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Create patch embeddings
        x = self.patch_embedding(x)  # (batch, num_patches, input_size, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        
        # Apply transformer layers
        for layer in self.layers:
            # Multi-head attention with residual
            attn_out = layer['attention'](x)
            x = layer['norm1'](x + attn_out)
            
            # FFN with residual
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # Use the last patch for prediction
        x = x[:, -1]  # (batch, input_size, d_model)
        
        # Average across input dimensions
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Final prediction
        return self.prediction_head(x)
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Return default parameters for the model"""
        return {
            # input_size will be set based on actual data
            'patch_len': 16,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'learning_rate': 1e-4
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'patch_len': (8, 32),
            'd_model': (8, 256),
            'n_heads': (2, 8),
            'n_layers': (2, 6),
            'dropout': (0.0, 0.5),
            'learning_rate': (1e-9, 1e-3)
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 