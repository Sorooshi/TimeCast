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
from .file_utils import create_experiment_directories, get_experiment_directory_name, create_directory_safely
from .visualization import save_training_plots
from .config_manager import save_hyperparameters


def save_results(
    model_name: str,
    data_name: str,
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    params: Dict[str, Any],
    mode: str = 'apply',
    experiment_description: str = None,
    test_data_name: str = None
):
    """Save training results, metrics, and predictions."""
    
    # Determine which directories we actually need based on mode
    if mode == 'tune':
        # For tune mode, only need hyperparameters and logs
        needed_dirs = ['hyperparameters', 'logs']
    elif mode == 'train_tuned' or mode == 'train_default':
        # For training modes, need weights, results, plots, history, metrics
        # (hyperparameters saved in tune mode, logs handled separately)
        needed_dirs = ['weights', 'results', 'plots', 'history', 'metrics']
    elif mode == 'predict':
        # For predict mode, only need results, predictions, and metrics
        needed_dirs = ['results', 'predictions', 'metrics']
    else:
        # For other modes, create all directories (fallback)
        needed_dirs = ['results', 'history', 'plots', 'predictions', 'metrics', 'hyperparameters', 'logs', 'weights']
    
    # Create only the necessary directories, including test_data_name for predict mode
    exp_subdir = get_experiment_directory_name(
        data_name, 
        experiment_description, 
        params.get('sequence_length'),
        test_data_name=test_data_name,
        mode=mode
    )
    
    directories = {}
    for dir_name in needed_dirs:
        dir_path = Path(dir_name.capitalize()) / model_name / mode / exp_subdir
        create_directory_safely(dir_path)
        directories[dir_name] = dir_path
    
    # Add experiment description to parameters
    params_with_exp = params.copy()
    params_with_exp['experiment_description'] = experiment_description
    
    # Save training history (only for modes that have history directory)
    if 'history' in directories:
        try:
            # Handle different types of history data structures
            if history and isinstance(history, dict):
                # Filter out non-numeric data (like metadata from k-fold)
                clean_history = {}
                for key, value in history.items():
                    if not key.startswith('_') and isinstance(value, (list, np.ndarray)):
                        # Ensure all values are lists and have the same length
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        clean_history[key] = value
                
                # Only create DataFrame if we have clean data
                if clean_history:
                    # Ensure all lists have the same length
                    lengths = [len(v) for v in clean_history.values()]
                    if len(set(lengths)) == 1:  # All same length
                        history_df = pd.DataFrame(clean_history)
                        history_df.to_csv(directories['history'] / "training_history.csv", index=False)
                        print(f"Saved training history to: {directories['history'] / 'training_history.csv'}")
                    else:
                        print(f"Warning: Training history data has inconsistent lengths - skipping CSV save")
                else:
                    print(f"Note: No suitable training history data to save (K-fold metadata only)")
            else:
                print(f"Note: No training history data to save")
        except Exception as e:
            print(f"Error saving training history: {e}")
            # Try to save whatever we can as JSON for debugging
            try:
                with open(directories['history'] / "training_history.json", "w") as f:
                    json.dump(history, f, indent=4, default=str)
                print(f"Saved training history as JSON for debugging: {directories['history'] / 'training_history.json'}")
            except:
                print(f"Could not save training history in any format")
    
    # Save training plots (only for modes that have plots directory)
    if 'plots' in directories:
        try:
            save_training_plots(history, directories['plots'], model_name)
        except Exception as e:
            print(f"Error saving training plots: {e}")
    
    # Save metrics (for modes that have metrics directory)
    if 'metrics' in directories:
        try:
            metrics_formatted = {k: f"{v:.4f}" for k, v in metrics.items()}
            with open(directories['metrics'] / "metrics.json", "w") as f:
                json.dump(metrics_formatted, f, indent=4)
            print(f"Saved metrics to: {directories['metrics'] / 'metrics.json'}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    # Save predictions and calculate per-sample metrics (for modes that have predictions directory)
    if 'predictions' in directories:
        try:
            # Check if test and validation predictions are different
            val_preds = predictions['val_predictions'].flatten()
            test_preds = predictions['test_predictions'].flatten() 
            val_targets = predictions['val_targets'].flatten()
            test_targets = predictions['test_targets'].flatten()
            
            # Always save validation predictions
            val_predictions_df = pd.DataFrame({
                'predictions': val_preds,
                'targets': val_targets
            })
            val_predictions_df['absolute_error'] = abs(val_predictions_df['predictions'] - val_predictions_df['targets'])
            val_predictions_df['squared_error'] = (val_predictions_df['predictions'] - val_predictions_df['targets']) ** 2
            # Calculate percentage error (avoiding division by zero)
            mask = val_predictions_df['targets'] != 0
            val_predictions_df['percentage_error'] = 0.0
            val_predictions_df.loc[mask, 'percentage_error'] = abs(
                (val_predictions_df.loc[mask, 'predictions'] - val_predictions_df.loc[mask, 'targets'])
                / val_predictions_df.loc[mask, 'targets'] * 100
            )
            val_predictions_df.to_csv(directories['predictions'] / "val_predictions.csv", index=False)
            
            # Only save test predictions if they're different from validation predictions
            if not (np.array_equal(val_preds, test_preds) and np.array_equal(val_targets, test_targets)):
                test_predictions_df = pd.DataFrame({
                    'predictions': test_preds,
                    'targets': test_targets
                })
                test_predictions_df['absolute_error'] = abs(test_predictions_df['predictions'] - test_predictions_df['targets'])
                test_predictions_df['squared_error'] = (test_predictions_df['predictions'] - test_predictions_df['targets']) ** 2
                # Calculate percentage error (avoiding division by zero)
                mask = test_predictions_df['targets'] != 0
                test_predictions_df['percentage_error'] = 0.0
                test_predictions_df.loc[mask, 'percentage_error'] = abs(
                    (test_predictions_df.loc[mask, 'predictions'] - test_predictions_df.loc[mask, 'targets'])
                    / test_predictions_df.loc[mask, 'targets'] * 100
                )
                test_predictions_df.to_csv(directories['predictions'] / "test_predictions.csv", index=False)
                print(f"Saved predictions to: {directories['predictions']}")
            else:
                print(f"Saved validation predictions to: {directories['predictions']}")
                print(f"Note: Test predictions identical to validation predictions - not duplicated")
                
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    # Save summary (for modes that have results directory)
    if 'results' in directories:
        try:
            # Build predictions file paths - test might not exist if identical to val
            val_preds = predictions['val_predictions'].flatten()
            test_preds = predictions['test_predictions'].flatten()
            val_targets = predictions['val_targets'].flatten()
            test_targets = predictions['test_targets'].flatten()
            
            predictions_files = {
                'val': str(directories['predictions'] / "val_predictions.csv") if 'predictions' in directories else "N/A"
            }
            
            # Only include test file if it was actually saved (different from val)
            if 'predictions' in directories and not (np.array_equal(val_preds, test_preds) and np.array_equal(val_targets, test_targets)):
                predictions_files['test'] = str(directories['predictions'] / "test_predictions.csv")
            else:
                predictions_files['test'] = "identical_to_validation" if 'predictions' in directories else "N/A"
            
            summary = {
                'experiment_description': experiment_description,
                'metrics': metrics_formatted if 'metrics' in directories else "N/A",
                'hyperparameters': params_with_exp,
                'files': {
                    'history': str(directories['history'] / "training_history.csv") if 'history' in directories else "N/A",
                    'predictions': predictions_files,
                    'metrics': str(directories['metrics'] / "metrics.json") if 'metrics' in directories else "N/A",
                    'hyperparameters': f"Hierarchical: Hyperparameters/{model_name}/{mode}/{exp_subdir}/" if 'hyperparameters' in directories else "N/A",
                    'plots': {
                        'loss': str(directories['plots'] / "loss_plot.png") if 'plots' in directories else "N/A",
                        'r2': str(directories['plots'] / "r2_plot.png") if 'plots' in directories else "N/A",
                        'mape': str(directories['plots'] / "mape_plot.png") if 'plots' in directories else "N/A"
                    }
                }
            }
            
            # Add test data name to summary if in predict mode
            if mode == 'predict' and test_data_name:
                summary['test_data_name'] = test_data_name
            
            with open(directories['results'] / "summary.json", "w") as f:
                json.dump(summary, f, indent=4)
            print(f"Saved summary to: {directories['results'] / 'summary.json'}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    # Print summary (only if we have metrics)
    if 'metrics' in directories or 'results' in directories:
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
    
    # Only print directories that exist in the dictionary
    if 'results' in directories:
        print(f"  Results: {directories['results']}")
    if 'history' in directories:
        print(f"  History: {directories['history']}")
    if 'predictions' in directories:
        print(f"  Predictions: {directories['predictions']}")
    if 'metrics' in directories:
        print(f"  Metrics: {directories['metrics']}")
    if 'hyperparameters' in directories:
        print(f"  Hyperparameters: {directories['hyperparameters']}")
    elif 'hyperparameters_path' in directories:
        print(f"  Hyperparameters: Used existing tuned parameters")
    if 'plots' in directories:
        print(f"  Plots: {directories['plots']}")


def load_and_print_results(model_name: str, data_name: str, mode: str, experiment_description: str = None, sequence_length: int = None):
    """Load and print results for a specific mode and experiment."""
    try:
        exp_subdir = get_experiment_directory_name(data_name, experiment_description, sequence_length)
        
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