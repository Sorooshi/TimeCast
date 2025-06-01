import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import torch

class TimeSeriesPreprocessor:
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        include_time_features: bool = True
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.include_time_features = include_time_features
        
    def create_sequences(
        self,
        data: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input data into sequences for time series prediction.
        
        Args:
            data: Shape (timesteps, N) where N is number of features (in our case number of merchants)
            dates: Optional datetime index for time features
            
        Returns:
            X: Shape (samples, sequence_length, features)
            y: Shape (samples, 1) - total consumption across all merchants
        """
        n_samples = len(data) - self.sequence_length
        
        # Create sequences
        X = np.zeros((n_samples, self.sequence_length, self.n_features))
        y = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            X[i] = data[i:i + self.sequence_length]
            # Sum across all merchants to get total consumption
            y[i] = np.sum(data[i + self.sequence_length])
            
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
    include_time_features: bool = True
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
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        input_size: Actual number of input features
    """
    # Calculate base input size from data
    base_input_size = data.shape[1]
    
    # Calculate additional features if time features are included
    time_features_size = 0
    if include_time_features:
        time_features_size = 1 + 7 + 1  # hour + day_of_week (one-hot) + is_holiday
    
    # Total input size
    input_size = base_input_size + time_features_size
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=sequence_length,
        n_features=base_input_size,
        include_time_features=include_time_features
    )
    
    X, y = preprocessor.create_sequences(data, dates)
    
    # Verify the input size matches our calculation
    assert X.shape[2] == input_size, f"Input size mismatch. Expected {input_size}, got {X.shape[2]}"
    
    # Calculate split indices
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split into train, validation, and test
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Convert to PyTorch tensors and create datasets
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
        batch_size=batch_size,
        shuffle=False  # No need to shuffle validation data
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )
    
    return train_loader, val_loader, test_loader, input_size 