"""
Results Manager
Handles saving and loading of training results, metrics, and predictions.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .file_utils import create_experiment_directories, get_experiment_directory_name
from .visualization import save_training_plots
from .config_manager import save_hyperparameters


def save_results(
    model_name: str,
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    params: Dict[str, Any],
    mode: str = 'apply',
    experiment_description: str = None
):
    """Save training results, metrics, and predictions."""
    # Create all necessary directories
    directories = create_experiment_directories(
        model_name, mode, experiment_description, params.get('sequence_length')
    )
    
    # Add experiment description to parameters
    params_with_exp = params.copy()
    params_with_exp['experiment_description'] = experiment_description
    
    # Save training history
    try:
        history_df = pd.DataFrame(history)
        history_df.to_csv(directories['history'] / "training_history.csv", index=False)
        print(f"Saved training history to: {directories['history'] / 'training_history.csv'}")
    except Exception as e:
        print(f"Error saving training history: {e}")
    
    # Save training plots
    try:
        save_training_plots(history, directories['plots'], model_name)
    except Exception as e:
        print(f"Error saving training plots: {e}")
    
    # Save metrics
    try:
        metrics_formatted = {k: f"{v:.4f}" for k, v in metrics.items()}
        with open(directories['metrics'] / "metrics.json", "w") as f:
            json.dump(metrics_formatted, f, indent=4)
        print(f"Saved metrics to: {directories['metrics'] / 'metrics.json'}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
    
    # Save predictions and calculate per-sample metrics
    try:
        for split in ["val", "test"]:
            predictions_df = pd.DataFrame({
                'predictions': predictions[f'{split}_predictions'].flatten(),
                'targets': predictions[f'{split}_targets'].flatten()
            })
            predictions_df['absolute_error'] = abs(
                predictions_df['predictions'] - predictions_df['targets']
            )
            predictions_df['squared_error'] = (
                predictions_df['predictions'] - predictions_df['targets']
            ) ** 2
            # Calculate percentage error (avoiding division by zero)
            mask = predictions_df['targets'] != 0
            predictions_df['percentage_error'] = 0.0
            predictions_df.loc[mask, 'percentage_error'] = abs(
                (predictions_df.loc[mask, 'predictions'] - predictions_df.loc[mask, 'targets'])
                / predictions_df.loc[mask, 'targets'] * 100
            )
            predictions_df.to_csv(directories['predictions'] / f"{split}_predictions.csv", index=False)
        print(f"Saved predictions to: {directories['predictions']}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    # Save hyperparameters
    save_hyperparameters(
        params_with_exp, 
        directories['hyperparams'], 
        mode, 
        is_tune_mode=(mode == 'tune')
    )
    
    # Save summary
    try:
        summary = {
            'experiment_description': experiment_description,
            'metrics': metrics_formatted,
            'hyperparameters': params_with_exp,
            'files': {
                'history': str(directories['history'] / "training_history.csv"),
                'predictions': {
                    'val': str(directories['predictions'] / "val_predictions.csv"),
                    'test': str(directories['predictions'] / "test_predictions.csv")
                },
                'metrics': str(directories['metrics'] / "metrics.json"),
                'hyperparameters': str(directories['hyperparams'] / f"{mode}_parameters.json"),
                'plots': {
                    'loss': str(directories['plots'] / "loss_plot.png"),
                    'r2': str(directories['plots'] / "r2_plot.png"),
                    'mape': str(directories['plots'] / "mape_plot.png")
                }
            }
        }
        
        with open(directories['results'] / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved summary to: {directories['results'] / 'summary.json'}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    # Print summary
    print_results_summary(metrics, experiment_description, directories)


def print_results_summary(
    metrics: Dict[str, float], 
    experiment_description: str, 
    directories: Dict[str, Path]
):
    """Print a formatted summary of results."""
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Experiment: {experiment_description or 'Default'}")
    print("Validation Metrics:")
    print(f"  Loss: {metrics['val_loss']:.4f}")
    print(f"  R² Score: {metrics['val_r2']:.4f}")
    print(f"  MAPE: {metrics['val_mape']:.2f}%")
    print("\nTest Metrics:")
    print(f"  Loss: {metrics['test_loss']:.4f}")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"  MAPE: {metrics['test_mape']:.2f}%")
    print("-" * 50)
    print(f"\nResults saved in:")
    print(f"  Results: {directories['results']}")
    print(f"  History: {directories['history']}")
    print(f"  Predictions: {directories['predictions']}")
    print(f"  Metrics: {directories['metrics']}")
    print(f"  Hyperparameters: {directories['hyperparams']}")
    print(f"  Plots: {directories['plots']}")


def load_and_print_results(model_name: str, mode: str, experiment_description: str = None, sequence_length: int = None):
    """Load and print results for a specific mode and experiment."""
    try:
        exp_subdir = get_experiment_directory_name(experiment_description, sequence_length)
        
        # Define directories
        base_dir = Path(".").resolve()
        metrics_dir = base_dir / "Metrics" / model_name / mode / exp_subdir
        hyperparams_dir = base_dir / "Hyperparameters" / model_name / exp_subdir  
        history_dir = base_dir / "History" / model_name / mode / exp_subdir
        
        # Load metrics
        with open(metrics_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
            
        # Load hyperparameters
        with open(hyperparams_dir / f"{mode}_parameters.json", "r") as f:
            params = json.load(f)
            
        # Load history
        history = pd.read_csv(history_dir / "training_history.csv")
        
        print(f"\n{mode.capitalize()} Mode Results:")
        print("-" * 50)
        
        print(f"Experiment: {params.get('experiment_description', 'Not specified')}")
        
        print("\nHyperparameters:")
        for param, value in params.items():
            if param != 'experiment_description':
                print(f"  {param}: {value}")
            
        print("\nFinal Metrics:")
        print("Validation:")
        print(f"  Loss: {metrics['val_loss']}")
        print(f"  R² Score: {metrics['val_r2']}")
        print(f"  MAPE: {metrics['val_mape']}%")
        
        print("Test:")
        print(f"  Loss: {metrics['test_loss']}")
        print(f"  R² Score: {metrics['test_r2']}")
        print(f"  MAPE: {metrics['test_mape']}%")
        
        print("\nTraining Summary:")
        print(f"  Best Validation Loss: {min(history['val_loss']):.4f}")
        print(f"  Best Validation R²: {max(history['val_r2']):.4f}")
        print(f"  Best Validation MAPE: {min(history['val_mape']):.2f}%")
        print(f"  Total Epochs: {len(history)}")
        
        return True
    except FileNotFoundError as e:
        print(f"No saved results found for {mode} mode with experiment '{experiment_description}': {str(e)}")
        return False 