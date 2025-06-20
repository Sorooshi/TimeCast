"""
Time Series Forecasting Package
Data preprocessing and loading utilities.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import torch


class TimeSeriesPreprocessor:
    def __init__(
        self,
        sequence_length: int,
        normalization: str = 'minmax'  # Options: 'standard', 'minmax', None
    ):
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.scalers = None  # Will store fitted scalers
        
    def fit_scalers(self, data: np.ndarray) -> None:
        """Fit scalers on training data."""
        if self.normalization is None:
            return
            
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Initialize scalers for each feature
        self.scalers = []
        for i in range(data.shape[1]):
            if self.normalization == 'standard':
                scaler = StandardScaler()
            elif self.normalization == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization}")
                
            # Reshape to 2D array for sklearn
            feature_data = data[:, i].reshape(-1, 1)
            scaler.fit(feature_data)
            self.scalers.append(scaler)
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self.normalization is None or self.scalers is None:
            return data
            
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            feature_data = data[:, i].reshape(-1, 1)
            normalized_data[:, i] = self.scalers[i].transform(feature_data).ravel()
            
        return normalized_data

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input data into sequences for time series prediction.
        
        Args:
            data: Shape (timesteps, N) where N is number of features
            
        Returns:
            X: Shape (samples, sequence_length, features)
            y: Shape (samples, 1) - total consumption across all merchants
        """
        # Normalize data if requested
        normalized_data = self.normalize_data(data)
        
        n_samples = len(normalized_data) - self.sequence_length
        n_features = normalized_data.shape[1]
        
        # Create sequences
        X = np.zeros((n_samples, self.sequence_length, n_features))
        y = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            X[i] = normalized_data[i:i + self.sequence_length]
            # Sum across all merchants to get total consumption
            y[i] = np.sum(data[i + self.sequence_length])  # Use original data for target
            
        return X, y


def prepare_data_for_model(
    data: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,  # Keep for compatibility but not used
    sequence_length: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    normalization: str = 'minmax'  # Options: 'standard', 'minmax', None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Prepare data loaders for training, validation, and test.
    
    Args:
        data: Input data of shape (timesteps, n_features)
        dates: Optional datetime index (kept for compatibility, not used)
        sequence_length: Length of input sequences
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for data loaders
        normalization: Type of normalization to apply ('standard', 'minmax', or None)
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        input_size: Number of input features (from original data)
    """
    # Calculate input size from data (no time features added)
    input_size = data.shape[1]
    print(f"Input size (from data): {input_size}")
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=sequence_length,
        normalization=normalization
    )
    
    # Calculate split indices
    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split raw data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Data splits - Train: {train_size}, Val: {val_size}, Test: {len(test_data)}")
    
    # Fit scalers on training data only
    preprocessor.fit_scalers(train_data)
    
    # Create sequences for each split using the same preprocessor
    X_train, y_train = preprocessor.create_sequences(train_data)
    X_val, y_val = preprocessor.create_sequences(val_data)
    X_test, y_test = preprocessor.create_sequences(test_data)
    
    print(f"Sequence shapes - X: {X_train.shape}, y: {y_train.shape}")
    
    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader, input_size 