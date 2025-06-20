"""
Workflow Manager
Handles different training workflows and mode logic.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import importlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from utils.training import tune_hyperparameters, TimeSeriesTrainer
from .config_manager import load_hyperparameters, filter_model_parameters
from .results_manager import save_results, load_and_print_results
from .file_utils import create_directory_safely


def load_model_class(model_name: str):
    """
    Dynamically load the model class.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model_class, actual_model_name)
    """
    try:
        models = importlib.import_module('models')
        model_class = getattr(models, model_name)
        actual_model_name = model_class.__name__
        return model_class, actual_model_name
    except (ImportError, AttributeError):
        raise ValueError(f"Model {model_name} not found. Available models: LSTM, TCN, Transformer, HybridTCNLSTM, PatchTST")


def setup_logging(model_name: str, mode: str) -> Path:
    """
    Set up logging for the experiment with actual file logging.
    
    Args:
        model_name: Name of the model
        mode: Training mode
        
    Returns:
        Path to logs directory
    """
    logs_dir = Path("Logs") / model_name
    create_directory_safely(logs_dir)
    
    if mode == 'tune':
        # Create a unique log file for this tuning session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"tuning_log_{timestamp}.txt"
        
        # Set up file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ],
            force=True  # Override any existing logging configuration
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        logger.info(f"Tuning logs will be saved to: {log_file}")
        
        print(f"\nTuning logs will be saved to: {log_file}")
    
    return logs_dir


def run_tune_mode(
    model_class, 
    model_name: str,
    train_loader, 
    val_loader, 
    test_loader,
    input_size: int,
    args
) -> None:
    """
    Run hyperparameter tuning mode with logging.
    
    Args:
        model_class: The model class
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        input_size: Input size for the model
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting hyperparameter tuning with {args.n_trials} trials")
    logger.info(f"Training parameters: epochs={args.epochs}, patience={args.patience}")
    logger.info(f"Data info: input_size={input_size}, sequence_length={args.sequence_length}")
    
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
    
    logger.info("Hyperparameter tuning completed")
    logger.info(f"Best validation loss: {tuning_metrics['val_loss']:.4f}")
    logger.info(f"Best validation R² score: {tuning_metrics['val_r2']:.4f}")
    logger.info(f"Best validation MAPE: {tuning_metrics['val_mape']:.2f}%")
    logger.info(f"Best parameters: {best_params}")
    
    print("\nTuning Results:")
    print(f"Best validation loss: {tuning_metrics['val_loss']:.4f}")
    print(f"Best validation R² score: {tuning_metrics['val_r2']:.4f}")
    print(f"Best validation MAPE: {tuning_metrics['val_mape']:.2f}%")
    
    # Train final model with best parameters
    best_params['input_size'] = input_size
    best_params['sequence_length'] = args.sequence_length
    
    logger.info("Training final model with best parameters")
    
    model_params = filter_model_parameters(best_params)
    model = model_class(**model_params)
    trainer = TimeSeriesTrainer(model)
    
    history, metrics, predictions = trainer.train_and_evaluate(
        train_loader,
        val_loader,
        test_loader,
        epochs=args.epochs,
        patience=args.patience,
        params=best_params
    )
    
    logger.info("Final model training completed")
    logger.info(f"Final test loss: {metrics['test_loss']:.4f}")
    logger.info(f"Final test R² score: {metrics['test_r2']:.4f}")
    logger.info(f"Final test MAPE: {metrics['test_mape']:.2f}%")
    
    experiment_desc = args.experiment_description or f"seq_len_{args.sequence_length}"
    save_results(model_name, history, metrics, predictions, best_params, 
                mode='tune', experiment_description=experiment_desc)
    
    logger.info(f"Results saved for experiment: {experiment_desc}")


def run_apply_mode(
    model_class,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    args,
    use_tuned: bool = True
) -> None:
    """
    Run apply mode (with or without tuned parameters).
    
    Args:
        model_class: The model class
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        input_size: Input size for the model
        args: Command line arguments
        use_tuned: Whether to use tuned parameters
    """
    # Load parameters
    params = load_hyperparameters(model_name, model_class, use_tuned=use_tuned)
    params['input_size'] = input_size
    params['sequence_length'] = args.sequence_length
    
    # Filter out non-model parameters
    model_params = filter_model_parameters(params)
    
    # Initialize and train model
    model = model_class(**model_params)
    trainer = TimeSeriesTrainer(model)
    
    history, metrics, predictions = trainer.train_and_evaluate(
        train_loader,
        val_loader,
        test_loader,
        epochs=args.epochs,
        patience=args.patience,
        params=params
    )
    
    # Determine mode name and experiment description
    mode = 'apply' if use_tuned else 'apply_not_tuned'
    experiment_desc = args.experiment_description or f"seq_len_{args.sequence_length}"
    
    save_results(model_name, history, metrics, predictions, params,
                mode=mode, experiment_description=experiment_desc)


def run_report_mode(args) -> None:
    """
    Run report mode to display saved results.
    
    Args:
        args: Command line arguments
    """
    print(f"\nReporting results for {args.model} on {args.data_name} dataset")
    print(f"Experiment: {args.experiment_description or f'seq_len_{args.sequence_length}'}")
    print("=" * 70)
    
    # Try to load results from different modes
    found_any = False
    modes_to_check = ['tune', 'apply', 'apply_not_tuned']
    
    for mode in modes_to_check:
        experiment_desc = args.experiment_description or f"seq_len_{args.sequence_length}"
        if load_and_print_results(args.model, mode, experiment_desc, args.sequence_length):
            found_any = True
            print("\n" + "=" * 70)
    
    if not found_any:
        print("No results found. Run the model in one of the available modes first.")
        print("Available modes: tune, apply, apply_not_tuned")


def get_mode_description(mode: str) -> str:
    """Get a human-readable description of the mode."""
    descriptions = {
        'tune': 'Hyperparameter tuning and training with best parameters',
        'apply': 'Training with previously tuned parameters (if available) or defaults',
        'apply_not_tuned': 'Training with default parameters only (ignoring tuned parameters)',
        'report': 'Display saved results from previous runs'
    }
    return descriptions.get(mode, 'Unknown mode') 