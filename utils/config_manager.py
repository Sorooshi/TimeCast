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


def load_hyperparameters(
    unique_specifier: str, 
    model_class, 
    use_tuned: bool = True
) -> Dict[str, Any]:
    """
    Load hyperparameters for the model using unique specifier.
    
    Args:
        unique_specifier: Unique identifier for the experiment
        model_class: The model class to get default parameters from
        use_tuned: Whether to use tuned parameters if available
        
    Returns:
        Dictionary of hyperparameters
    """
    if use_tuned:
        hyperparams_dir = Path("Hyperparameters")
        tuned_params_path = hyperparams_dir / f"{unique_specifier}_tuned.json"
        
        if tuned_params_path.exists():
            try:
                with open(tuned_params_path, "r") as f:
                    params = json.load(f)
                print(f"\nUsing previously tuned hyperparameters for {unique_specifier}")
                return params
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"\nError loading tuned parameters: {e}")
                print("Falling back to default parameters")
        else:
            print(f"\nNo tuned parameters found for {unique_specifier}, using default parameters")
    else:
        print(f"\nUsing default parameters for {unique_specifier} (not tuned)")
    
    return model_class.get_default_parameters()


def save_hyperparameters_with_specifier(
    params: Dict[str, Any], 
    unique_specifier: str, 
    mode: str
) -> None:
    """
    Save hyperparameters using unique specifier.
    
    Args:
        params: Parameters to save
        unique_specifier: Unique identifier for the experiment
        mode: Mode (tune, train, apply_not_tuned, predict)
    """
    try:
        hyperparams_dir = Path("Hyperparameters")
        hyperparams_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == 'tune':
            # Save as tuned parameters for this specifier
            file_path = hyperparams_dir / f"{unique_specifier}_tuned.json"
        else:
            # Save as mode-specific parameters
            file_path = hyperparams_dir / f"{unique_specifier}_{mode}.json"
        
        with open(file_path, "w") as f:
            json.dump(params, f, indent=4)
        
        print(f"Saved hyperparameters to: {file_path}")
    except Exception as e:
        print(f"Error saving hyperparameters: {e}")


def load_model_weights(unique_specifier: str, use_tuned: bool = True) -> Path:
    """
    Get the path to the model weights file.
    
    Args:
        unique_specifier: Unique identifier for the experiment
        use_tuned: Whether to use tuned model weights
        
    Returns:
        Path to the weights file
    """
    weights_dir = Path("Weights")
    
    if use_tuned:
        weights_path = weights_dir / f"{unique_specifier}_tuned_best.pth"
    else:
        weights_path = weights_dir / f"{unique_specifier}_default_best.pth"
    
    return weights_path


def save_model_weights(model, unique_specifier: str, use_tuned: bool = True) -> None:
    """
    Save model weights using unique specifier.
    
    Args:
        model: The trained model
        unique_specifier: Unique identifier for the experiment
        use_tuned: Whether these are tuned model weights
    """
    import torch
    
    weights_dir = Path("Weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    if use_tuned:
        weights_path = weights_dir / f"{unique_specifier}_tuned_best.pth"
    else:
        weights_path = weights_dir / f"{unique_specifier}_default_best.pth"
    
    try:
        torch.save(model.state_dict(), weights_path)
        print(f"Saved model weights to: {weights_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")


def filter_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out non-model parameters before creating the model.
    Only includes sequence_length for models that need it (like MLP).
    
    Args:
        params: Dictionary of all parameters
        
    Returns:
        Dictionary with only model parameters
    """
    # Get the model class name from the call stack to determine which parameters to include
    import inspect
    frame = inspect.currentframe()
    try:
        # Look through the call stack to find model class information
        for f in inspect.getouterframes(frame):
            if 'model_class' in f.frame.f_locals:
                model_class = f.frame.f_locals['model_class']
                model_name = model_class.__name__ if hasattr(model_class, '__name__') else str(model_class)
                
                # Only include sequence_length for models that need it
                if model_name == 'MLP':
                    return {k: v for k, v in params.items() 
                            if k not in ['experiment_description']}
                else:
                    return {k: v for k, v in params.items() 
                            if k not in ['sequence_length', 'experiment_description']}
    finally:
        del frame
    
    # Fallback: exclude sequence_length by default for safety
    return {k: v for k, v in params.items() 
            if k not in ['sequence_length', 'experiment_description']}


# Legacy function for backwards compatibility
def save_hyperparameters(
    params: Dict[str, Any], 
    hyperparams_dir: Path, 
    mode: str,
    is_tune_mode: bool = False
) -> None:
    """
    Legacy save hyperparameters function - kept for backwards compatibility.
    
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