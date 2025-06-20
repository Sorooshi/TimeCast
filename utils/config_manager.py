"""
Configuration Manager
Handles hyperparameter loading, saving, and management.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_hyperparameters(model_name: str, model_class, use_tuned: bool = True) -> Dict[str, Any]:
    """
    Load hyperparameters for the model.
    
    Args:
        model_name: Name of the model
        model_class: The model class to get default parameters from
        use_tuned: Whether to use tuned parameters if available
        
    Returns:
        Dictionary of hyperparameters
    """
    if use_tuned:
        hyperparams_dir = Path("Hyperparameters") / model_name
        tuned_params_path = hyperparams_dir / "tuned_parameters.json"
        
        if tuned_params_path.exists():
            try:
                with open(tuned_params_path, "r") as f:
                    params = json.load(f)
                print("\nUsing previously tuned hyperparameters")
                return params
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"\nError loading tuned parameters: {e}")
                print("Falling back to default parameters")
        else:
            print("\nNo tuned parameters found, using default parameters")
    else:
        print("\nUsing default parameters (not tuned)")
    
    return model_class.get_default_parameters()


def filter_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out non-model parameters before creating the model.
    
    Args:
        params: Dictionary of all parameters
        
    Returns:
        Dictionary with only model parameters
    """
    return {k: v for k, v in params.items() 
            if k not in ['sequence_length', 'experiment_description']}


def save_hyperparameters(
    params: Dict[str, Any], 
    hyperparams_dir: Path, 
    mode: str,
    is_tune_mode: bool = False
) -> None:
    """
    Save hyperparameters to files.
    
    Args:
        params: Parameters to save
        hyperparams_dir: Directory to save parameters
        mode: Mode (tune, apply, etc.)
        is_tune_mode: Whether this is from tuning mode
    """
    try:
        # Save mode-specific parameters
        with open(hyperparams_dir / f"{mode}_parameters.json", "w") as f:
            json.dump(params, f, indent=4)
        
        # If in tune mode, also save as the main tuned parameters
        if is_tune_mode:
            # Save tuned parameters at model level (without experiment subdirectory)
            main_hyperparams_dir = hyperparams_dir.parent
            from .file_utils import create_directory_safely
            create_directory_safely(main_hyperparams_dir)
            with open(main_hyperparams_dir / "tuned_parameters.json", "w") as f:
                json.dump(params, f, indent=4)
        
        print(f"Saved hyperparameters to: {hyperparams_dir}")
    except Exception as e:
        print(f"Error saving hyperparameters: {e}") 