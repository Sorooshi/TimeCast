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
    test_data: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,  # Keep for compatibility but not used
    sequence_length: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    normalization: str = 'minmax',  # Options: 'standard', 'minmax', None
    mode: str = 'tune'
):
    """
    Prepare data loaders for different modes.
    
    Args:
        data: Input data of shape (timesteps, n_features)
        test_data: Test data for predict mode (should be None for tune/train modes)
        dates: Optional datetime index (kept for compatibility, not used)
        sequence_length: Length of input sequences
        train_ratio: Proportion of data to use for training (only for tune/train modes)
        val_ratio: Proportion of data to use for validation (only for tune/train modes)
        batch_size: Batch size for data loaders
        normalization: Type of normalization to apply ('standard', 'minmax', or None)
        mode: Mode of operation ('tune', 'train', or 'predict')
        
    Returns:
        tune/train modes: (train_loader, val_loader, input_size)
        predict mode: (test_loader, input_size)
    """
    # Calculate input size from data (features only, excluding target)
    input_size = data.shape[1] - 1  # Exclude target column
    print(f"Input size (features only): {input_size}")
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=sequence_length,
        normalization=normalization
    )
    
    if mode in ['tune', 'train']:
        # Split data into train and validation only
        n_samples = len(data)
        train_size = int(n_samples * train_ratio)
        val_size = n_samples - train_size  # Use remaining data for validation
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        print(f"Data splits - Train: {train_size}, Val: {val_size}")
        
        # Fit scalers on training data only
        preprocessor.fit_scalers(train_data)
        
        # Create sequences
        X_train, y_train = preprocessor.create_sequences(train_data)
        X_val, y_val = preprocessor.create_sequences(val_data)
        
        print(f"Sequence shapes - Train X: {X_train.shape}, y: {y_train.shape}")
        print(f"Sequence shapes - Val X: {X_val.shape}, y: {y_val.shape}")
        
        # Create PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
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
        
        return train_loader, val_loader, input_size
        
    elif mode == 'predict':
        # Use original training data to fit scalers, then transform test data
        if test_data is None:
            raise ValueError("test_data must be provided for predict mode")
            
        print(f"Data splits - Training data: {len(data)}, Test data: {len(test_data)}")
        
        # Fit scalers on the original training data (same data used for training)
        preprocessor.fit_scalers(data)
        
        # Transform and create sequences from test data
        X_test, y_test = preprocessor.create_sequences(test_data)
        
        print(f"Sequence shapes - Test X: {X_test.shape}, y: {y_test.shape}")
        
        # Create PyTorch dataset
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        
        return test_loader, input_size
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'tune', 'train', or 'predict'") 