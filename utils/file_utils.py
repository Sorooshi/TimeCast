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


def get_experiment_directory_name(experiment_description: str = None, sequence_length: int = None) -> str:
    """
    Generate a safe directory name for experiments.
    
    Args:
        experiment_description: Custom experiment description
        sequence_length: Sequence length to use as fallback
        
    Returns:
        Safe directory name string
    """
    if experiment_description:
        exp_name = experiment_description
    else:
        exp_name = f"seq_len_{sequence_length or 'unknown'}"
    
    # Replace spaces and special characters with underscores for safe directory names
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in exp_name)


def create_experiment_directories(
    model_name: str, 
    mode: str, 
    experiment_description: str = None,
    sequence_length: int = None
) -> Dict[str, Path]:
    """
    Create all necessary directories for an experiment.
    
    Args:
        model_name: Name of the model
        mode: Mode (tune, apply, apply_not_tuned, etc.)
        experiment_description: Custom experiment description
        sequence_length: Sequence length for default naming
        
    Returns:
        Dictionary of directory paths
    """
    exp_subdir = get_experiment_directory_name(experiment_description, sequence_length)
    
    # Create base directories - all at the same level as main.py
    base_dir = Path(".").resolve()
    
    directories = {
        'results': base_dir / "Results" / model_name / mode / exp_subdir,
        'hyperparams': base_dir / "Hyperparameters" / model_name / exp_subdir,
        'predictions': base_dir / "Predictions" / model_name / mode / exp_subdir,
        'metrics': base_dir / "Metrics" / model_name / mode / exp_subdir,
        'history': base_dir / "History" / model_name / mode / exp_subdir,
        'plots': base_dir / "Plots" / model_name / mode / exp_subdir,
        'logs': base_dir / "Logs" / model_name / mode / exp_subdir,
        'weights': base_dir / "Weights" / model_name / mode / exp_subdir
    }
    
    # Create all directories with robust error handling
    failed_dirs = []
    for dir_name, directory in directories.items():
        if not create_directory_safely(directory):
            failed_dirs.append(f"{dir_name}: {directory}")
    
    if failed_dirs:
        print(f"Warning: Failed to create some directories: {failed_dirs}")
        print("Some results may not be saved properly.")
    
    return directories 