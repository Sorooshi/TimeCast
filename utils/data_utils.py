"""
Data Utilities
Handles data loading, validation, and preparation.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Any
from utils.data_preprocessing import prepare_data_for_model


def load_and_validate_data(data_path: Path) -> Tuple[Any, Any]:
    """
    Load and validate data from the specified path.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple of (data_array, dates_series)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data loading fails
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            f"Please ensure the file exists or provide the correct path using --data_path"
        )

    print(f"\nLoading data from: {data_path}")
    
    try:
        # Attempt to load the data
        df = pd.read_csv(data_path)
        
        if 'date' in df.columns or 'timestamp' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            dates = pd.to_datetime(df[date_col])
            data = df.drop(columns=[date_col]).values
        else:
            print("Warning: No date/timestamp column found. Using index as timeline.")
            dates = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            data = df.values
            
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

    print(f"Data shape: {data.shape}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    return data, dates


def get_data_path(data_name: str, data_path: str = None) -> Path:
    """
    Determine the correct data path.
    
    Args:
        data_name: Name of the dataset (without .csv extension)
        data_path: Full path to the data file (optional)
        
    Returns:
        Path object to the data file
    """
    if data_path is None:
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return data_dir / f"{data_name}.csv"
    else:
        return Path(data_path)


def prepare_data_loaders(data, dates, sequence_length: int):
    """
    [Depracated] Prepare data loaders for training.
    
    Args:
        data: Data array
        dates: Dates series
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_size)
    """
    return prepare_data_for_model(
        data=data,
        dates=dates,
        sequence_length=sequence_length
    ) 