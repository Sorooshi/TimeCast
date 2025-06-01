"""
Time Series Forecasting Package
Data preprocessing and loading utilities.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import torch

class TimeSeriesPreprocessor:
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        include_time_features: bool = True,
        normalization: str = 'standard'  # Options: 'standard', 'minmax', None
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.include_time_features = include_time_features
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
    
    def inverse_normalize(self, data: np.ndarray, feature_idx: int = None) -> np.ndarray:
        """Inverse transform normalized data for specified feature(s)."""
        if self.normalization is None or self.scalers is None:
            return data
            
        if feature_idx is None:
            # Inverse transform all features
            original_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                feature_data = data[:, i].reshape(-1, 1)
                original_data[:, i] = self.scalers[i].inverse_transform(feature_data).ravel()
            return original_data
        else:
            # Inverse transform single feature
            feature_data = data.reshape(-1, 1)
            return self.scalers[feature_idx].inverse_transform(feature_data).ravel()
        
    def create_sequences(
        self,
        data: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input data into sequences for time series prediction.
        
        Args:
            data: Shape (timesteps, N) where N is number of features
            dates: Optional datetime index for time features
            
        Returns:
            X: Shape (samples, sequence_length, features)
            y: Shape (samples, 1) - total consumption across all merchants
        """
        # Normalize data if requested
        normalized_data = self.normalize_data(data)
        
        n_samples = len(normalized_data) - self.sequence_length
        
        # Create sequences
        X = np.zeros((n_samples, self.sequence_length, self.n_features))
        y = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            X[i] = normalized_data[i:i + self.sequence_length]
            # Sum across all merchants to get total consumption
            y[i] = np.sum(data[i + self.sequence_length])  # Use original data for target
            
        if self.include_time_features and dates is not None:
            time_features = self._create_time_features(
                dates[self.sequence_length:]
            )
            X = np.concatenate([
                X,
                np.repeat(
                    time_features[:, np.newaxis, :],
                    self.sequence_length,
                    axis=1
                )
            ], axis=2)
            
        return X, y
    
    def _create_time_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Create time-based features."""
        # Convert to DatetimeIndex if not already
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
            
        # Hour of day (normalized to [0, 1])
        hour = dates.hour.values / 23.0
        
        # Day of week (one-hot encoded)
        dow = pd.get_dummies(dates.dayofweek).values
        
        # Is holiday (dummy for now - should be replaced with actual holiday data)
        is_holiday = np.zeros(len(dates))
        
        return np.column_stack([hour, dow, is_holiday])

def prepare_data_for_model(
    data: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    sequence_length: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    include_time_features: bool = True,
    normalization: str = 'minmax'  # Options: 'standard', 'minmax', None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Prepare data loaders for training, validation, and test.
    
    Args:
        data: Input data of shape (timesteps, n_merchants)
        dates: Optional datetime index
        sequence_length: Length of input sequences
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for data loaders
        include_time_features: Whether to include time-based features
        normalization: Type of normalization to apply ('standard', 'minmax', or None)
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        input_size: Actual number of input features
    """
    # Calculate base input size from data
    base_input_size = data.shape[1]
    
    # Calculate additional features if time features are included (can be modified)
    time_features_size = 0
    if include_time_features:
        print(f"include_time_features is True. If you have more than nine features, modify the line below.")
        time_features_size = 1 + 7 + 1  # hour + day_of_week (one-hot) + is_holiday
        
    
    # Total input size
    input_size = base_input_size + time_features_size
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=sequence_length,
        n_features=base_input_size,
        include_time_features=include_time_features,
        normalization=normalization
    )
    
    # Calculate split indices
    n_samples = len(data)  # Use original data length for splitting
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split raw data first
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Split dates if provided
    train_dates = None
    val_dates = None
    test_dates = None
    if dates is not None:
        train_dates = dates[:train_size]
        val_dates = dates[train_size:train_size + val_size]
        test_dates = dates[train_size + val_size:]
    
    # Fit scalers on training data only
    preprocessor.fit_scalers(train_data)
    
    # Create sequences for each split using the same preprocessor
    X_train, y_train = preprocessor.create_sequences(train_data, train_dates)
    X_val, y_val = preprocessor.create_sequences(val_data, val_dates)
    X_test, y_test = preprocessor.create_sequences(test_data, test_dates)
    
    # Verify the input size matches our calculation
    assert X_train.shape[2] == input_size, f"Input size mismatch. Expected {input_size}, got {X_train.shape[2]}"
    
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