"""
File and Directory Utilities
Handles file operations, directory creation, and path management.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import os
from pathlib import Path
from typing import Dict, Any
import re


def create_unique_specifier(
    model_name: str,
    data_name: str,
    sequence_length: int,
    experiment_description: str = None
) -> str:
    """
    Create a unique specifier for experiments.
    
    Args:
        model_name: Name of the model
        data_name: Name of the dataset
        sequence_length: Sequence length
        experiment_description: Custom experiment description
        
    Returns:
        Unique specifier string
    """
    
    if experiment_description is None:
        exp_desc = f"No_Description"
    else:
        exp_desc = experiment_description
    
    # Create unique specifier
    specifier = f"{model_name}_{data_name}_{exp_desc}_{sequence_length}"
    
    # Replace spaces and special characters with underscores for safe file/directory names
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in specifier)


def create_directory_safely(directory: Path) -> bool:
    """
    Safely create a directory with proper error handling and validation.
    
    Args:
        directory: Path object for the directory to create
        
    Returns:
        bool: True if directory exists and is accessible, False otherwise
    """
    try:
        # Create directory with parents
        directory.mkdir(parents=True, exist_ok=True)
        
        # Verify directory exists
        if not directory.exists():
            print(f"Warning: Failed to create directory: {directory}")
            return False
            
        # Check if directory is accessible
        if not os.access(directory, os.R_OK | os.W_OK):
            print(f"Warning: Directory is not accessible: {directory}")
            return False
            
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def get_experiment_directory_name(
    data_name: str,
    experiment_description: str = None,
    sequence_length: int = None,
    test_data_name: str = None,
    mode: str = None
) -> str:
    """
    Create a standardized directory name for experiments.
    
    Args:
        data_name: Name of the dataset
        experiment_description: Custom experiment description
        sequence_length: Sequence length
        test_data_name: Name of test dataset (only used in predict mode)
        mode: Mode of operation (tune, train, predict, etc.)
        
    Returns:
        Directory name string
    """
    parts = []
    
    # Add sequence length if provided
    if sequence_length is not None:
        parts.append(f"seq_len_{sequence_length}")
    
    # Add experiment description if provided
    if experiment_description:
        # Clean up experiment description - replace spaces and special chars with underscores
        clean_desc = re.sub(r'[^\w\s-]', '', experiment_description)
        clean_desc = re.sub(r'[-\s]+', '_', clean_desc).strip('-_')
        parts.append(clean_desc)
    
    # For predict mode, add test dataset name if different from training data
    if mode == 'predict' and test_data_name and test_data_name != data_name:
        parts.append(f"test_{test_data_name}")
    
    return '/'.join(parts) if parts else "default"


def create_experiment_directories(
    model_name: str, 
    data_name: str,
    mode: str, 
    experiment_description: str = None,
    sequence_length: int = None
) -> Dict[str, Path]:
    """
    Create all necessary directories for an experiment.
    
    Args:
        model_name: Name of the model
        data_name: Name of the dataset
        mode: Mode (tune, train_default, train_tuned, predict, etc.)
        experiment_description: Custom experiment description
        sequence_length: Sequence length for default naming
        
    Returns:
        Dictionary of directory paths
    """
    exp_subdir = get_experiment_directory_name(data_name, experiment_description, sequence_length)
    
    directories = {}
    
    # Create hierarchical directory structure for each file type
    base_dirs = ['Results', 'History', 'Plots', 'Predictions', 'Metrics', 'Weights', 'Hyperparameters', 'Logs']
    
    for base_dir in base_dirs:
        dir_path = Path(base_dir) / model_name / mode / exp_subdir
        create_directory_safely(dir_path)
        directories[base_dir.lower()] = dir_path
    
    return directories 