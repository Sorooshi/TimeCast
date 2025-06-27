"""
TimeCast
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
        self.scalers = None  # Will store fitted scalers for features
        self.target_scaler = None  # Will store scaler for targets
        
    def fit_scalers(self, data: np.ndarray) -> None:
        """Fit scalers on training data (features and target separately)."""
        if self.normalization is None:
            return
            
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Initialize scalers for each feature (excluding target column)
        self.scalers = []
        for i in range(data.shape[1] - 1):  # Exclude last column (target)
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
        
        # Fit target scaler on last column only
        target_values = data[:, -1].reshape(-1, 1)
        if self.normalization == 'standard':
            self.target_scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.target_scaler = MinMaxScaler()
        
        self.target_scaler.fit(target_values)
        
    def normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """Apply normalization to target values."""
        if self.normalization is None or self.target_scaler is None:
            return targets
            
        return self.target_scaler.transform(targets.reshape(-1, 1)).ravel()
        
    def denormalize_targets(self, normalized_targets: np.ndarray) -> np.ndarray:
        """Convert normalized targets back to original scale."""
        if self.normalization is None or self.target_scaler is None:
            return normalized_targets
            
        return self.target_scaler.inverse_transform(normalized_targets.reshape(-1, 1)).ravel()

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to feature data only (excluding target column)."""
        if self.normalization is None or self.scalers is None:
            return data
            
        # Only normalize features (exclude last column which is the target)
        feature_data = data[:, :-1]  # All columns except last
        normalized_features = np.zeros_like(feature_data)
        
        for i in range(feature_data.shape[1]):
            feature_column = feature_data[:, i].reshape(-1, 1)
            normalized_features[:, i] = self.scalers[i].transform(feature_column).ravel()
            
        return normalized_features

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input data into sequences for time series prediction.
        
        Args:
            data: Shape (timesteps, N) where N is number of features (including target as last column)
            
        Returns:
            X: Shape (samples, sequence_length, features) - normalized features only
            y: Shape (samples, 1) - normalized target values
        """
        # Normalize feature data only (excludes target column)
        normalized_features = self.normalize_data(data)
        
        n_samples = len(normalized_features) - self.sequence_length
        n_features = normalized_features.shape[1]  # Features only (no target)
        
        # Create sequences
        X = np.zeros((n_samples, self.sequence_length, n_features))
        y = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            # Use normalized features for input sequences
            X[i] = normalized_features[i:i + self.sequence_length]
            
            # Get raw target from original data and normalize separately
            raw_target = data[i + self.sequence_length, -1]  # Last column only
            y[i] = self.normalize_targets(np.array([raw_target]))[0]
            
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
    # Calculate input size from data (features only, excluding target)
    input_size = data.shape[1] - 1  # Exclude target column
    print(f"Input size (features only): {input_size}")
    
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