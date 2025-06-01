import argparse
import pandas as pd
import torch
import numpy as np
from utils.training import tune_hyperparameters, TimeSeriesTrainer
from typing import Dict, Any, List
import importlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_hyperparameters(model_name: str, model_class) -> Dict[str, Any]:
    """
    Load hyperparameters for the model, preferring tuned parameters if available.
    
    Args:
        model_name: Name of the model/algorithm
        model_class: The model class to get default parameters from
        
    Returns:
        Dictionary of hyperparameters
    """
    hyperparams_dir = Path("hyperparameters") / model_name
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
    
    return model_class().get_default_parameters()

def save_training_plots(history: Dict[str, List[float]], save_dir: Path, model_name: str):
    """
    Save training and validation loss/MSE plots.
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save the plots
        model_name: Name of the model for plot titles
    """
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Create plots directory if it doesn't exist
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_mse'], label='Training MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.title(f'{model_name} - Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'mse_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot R² Score
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_r2'], label='Training R²')
    plt.plot(history['val_r2'], label='Validation R²')
    plt.title(f'{model_name} - Training and Validation R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'r2_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MAPE
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_mape'], label='Training MAPE')
    plt.plot(history['val_mape'], label='Validation MAPE')
    plt.title(f'{model_name} - Training and Validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'mape_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(
    model_name: str,
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    params: Dict[str, Any],
    mode: str = 'apply'
):
    """Save training results, metrics, and predictions."""
    # Create base directories if they don't exist
    base_dir = Path(".")
    results_dir = base_dir / "results" / model_name / mode
    hyperparams_dir = base_dir / "hyperparameters" / model_name
    predictions_dir = base_dir / "predictions" / model_name / mode
    metrics_dir = base_dir / "metrics" / model_name / mode
    history_dir = base_dir / "history" / model_name / mode
    plots_dir = base_dir / "plots" / model_name / mode
    
    # Create all directories
    for directory in [results_dir, hyperparams_dir, predictions_dir,
                       metrics_dir, history_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_dir / "training_history.csv", index=False)
    
    # Save training plots
    save_training_plots(history, results_dir, model_name)
    
    # Save metrics with more detailed formatting
    metrics_formatted = {
        k: f"{v:.4f}" for k, v in metrics.items()
    }
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics_formatted, f, indent=4)
    
    # Save predictions and calculate per-sample metrics
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
        predictions_df.to_csv(predictions_dir / f"{split}_predictions.csv", index=False)
    
    # Save hyperparameters in mode-specific directory
    with open(hyperparams_dir / f"{mode}_parameters.json", "w") as f:
        json.dump(params, f, indent=4)
    
    # If in tune mode, also save as the main tuned parameters
    if mode == 'tune':
        with open(hyperparams_dir / "tuned_parameters.json", "w") as f:
            json.dump(params, f, indent=4)
    
    # Save a summary of all results
    summary = {
        'metrics': metrics_formatted,
        'hyperparameters': params,
        'files': {
            'history': str(history_dir / "training_history.csv"),
            'predictions': {
                'val': str(predictions_dir / "val_predictions.csv"),
                'test': str(predictions_dir / "test_predictions.csv")
            },
            'metrics': str(metrics_dir / "metrics.json"),
            'hyperparameters': str(hyperparams_dir / f"{mode}_parameters.json"),
            'plots': {
                'loss': str(plots_dir / "loss_plot.png"),
                'mse': str(plots_dir / "mse_plot.png"),
                'r2': str(plots_dir / "r2_plot.png"),
                'mape': str(plots_dir / "mape_plot.png")
            }
        }
    }
    
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
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
    print(f"  Results: {results_dir}")
    print(f"  History: {history_dir}")
    print(f"  Predictions: {predictions_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Hyperparameters: {hyperparams_dir}")
    print(f"  Plots: {plots_dir}")

def load_and_print_results(model_name: str, mode: str):
    """Load and print results for a specific mode (tune or apply)."""
    try:
        # Define directories
        base_dir = Path(".")
        metrics_dir = base_dir / "metrics" / model_name / mode
        hyperparams_dir = base_dir / "hyperparameters" / model_name
        history_dir = base_dir / "history" / model_name / mode
        
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
        
        print("Hyperparameters:")
        for param, value in params.items():
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
        print(f"No saved results found for {mode} mode: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting with PyTorch')
    
    parser.add_argument('--algorithm', type=str, required=True,
                      help='Name of the algorithm to use')
    
    parser.add_argument('--data_name', type=str, required=True,
                      help='Name of the dataset to use (without .csv extension)')
    
    parser.add_argument('--data_path', type=str,
                      help='Full path to the data file. If not provided, '
                      'will look for {data_name}.csv in .data/ directory')
    
    parser.add_argument('--mode', type=str, choices=['tune', 'apply', 'report'],
                      default='apply', help='Mode to run the model in')
    
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of trials for hyperparameter tuning')
    
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    
    parser.add_argument('--patience', type=int, default=25,
                      help='Patience for early stopping')
    
    parser.add_argument('--sequence_length', type=int, default=10,
                      help='Length of input sequences')
    
    args = parser.parse_args()

    # Create logs directory for tuning
    logs_dir = Path("logs") / args.algorithm
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging for hyperparameter tuning
    if args.mode == 'tune':
        log_file = logs_dir / f"tuning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\nTuning logs will be saved to: {log_file}")
        
    if args.mode == 'report':
        print(f"\nReporting results for {args.algorithm} on {args.data_name} dataset")
        print("=" * 70)
        
        # Try to load both tuned and applied results
        found_any = False
        for mode in ['tune', 'apply']:
            if load_and_print_results(args.algorithm, mode):
                found_any = True
                print("\n" + "=" * 70)
        
        if not found_any:
            print("No results found. Run the model in 'tune' or 'apply' mode first.")
        return

    # Dynamically import the model class
    try:
        model_module = importlib.import_module(f'models.{args.algorithm.lower()}')
        model_class = getattr(model_module, args.algorithm)
    except (ImportError, AttributeError):
        raise ValueError(f"Model {args.algorithm} not found")

    # Determine data path and load data
    if args.data_path is None:
        # Create .data directory if it doesn't exist
        data_dir = Path(".data")
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir / f"{args.data_name}.csv"
    else:
        data_path = Path(args.data_path)

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

    # Prepare data loaders
    from utils.data_preprocessing import prepare_data_for_model
    train_loader, val_loader, test_loader, input_size = prepare_data_for_model(
        data=data,
        dates=dates,
        sequence_length=args.sequence_length
    )

    if args.mode == 'tune':
        # Perform hyperparameter tuning
        best_params, tuning_metrics = tune_hyperparameters(
            model_class,
            train_loader,
            val_loader,
            n_trials=args.n_trials,
            epochs=args.epochs,
            patience=args.patience,
            input_size=input_size
        )
        print("\nTuning Results:")
        print(f"Best validation loss: {tuning_metrics['val_loss']:.4f}")
        print(f"Best validation R² score: {tuning_metrics['val_r2']:.4f}")
        print(f"Best validation MAPE: {tuning_metrics['val_mape']:.2f}%")
        
        # Train final model with best parameters
        best_params['input_size'] = input_size
        model = model_class(**best_params)
        trainer = TimeSeriesTrainer(model)
        history, metrics, predictions = trainer.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            epochs=args.epochs,
            patience=args.patience,
            params=best_params
        )
        save_results(args.algorithm, history, metrics, predictions, best_params, mode='tune')
        
    else:  # apply mode
        # Load best parameters if available, otherwise use defaults
        params = load_hyperparameters(args.algorithm, model_class)
        params['input_size'] = input_size
        
        # Initialize and train model
        model = model_class(**params)
        trainer = TimeSeriesTrainer(model)
        history, metrics, predictions = trainer.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            epochs=args.epochs,
            patience=args.patience,
            params=params
        )
        save_results(
            args.algorithm,
            history,
            metrics,
            predictions,
            params,
            mode='apply'
        )
    
    print(f"\nDetailed results saved in results/{args.algorithm}/")

if __name__ == '__main__':
    main() 