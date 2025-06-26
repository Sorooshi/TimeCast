"""
Workflow Manager
Handles different training workflows and mode logic.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import importlib
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.model_selection import KFold
from utils.training import tune_hyperparameters, TimeSeriesTrainer
from .config_manager import (
    load_hyperparameters, filter_model_parameters, 
    save_hyperparameters_with_specifier, save_model_weights, load_model_weights
)
from .results_manager import save_results, load_and_print_results
from .file_utils import create_directory_safely
from .data_preprocessing import prepare_data_for_model


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
        raise ValueError(f"Model {model_name} not found. Available models: LSTM, TCN, Transformer, HybridTCNLSTM, MLP, ProphetModel")


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
    
    # Create a unique log file for this session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"{mode}_log_{timestamp}.txt"
    
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
    logger.info(f"Starting {mode} mode for {model_name}")
    logger.info(f"Logs will be saved to: {log_file}")
    
    print(f"\nLogs will be saved to: {log_file}")
    
    return logs_dir


def run_tune_mode(
    model_class, 
    model_name: str,
    unique_specifier: str,
    train_loader, 
    val_loader, 
    test_loader,
    input_size: int,
    args
) -> None:
    """
    Run hyperparameter tuning mode with logging.
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
        input_size=input_size,
        sequence_length=args.sequence_length
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
    
    # Save tuned hyperparameters with unique specifier
    best_params['input_size'] = input_size
    best_params['sequence_length'] = args.sequence_length
    best_params['experiment_description'] = args.experiment_description
    
    save_hyperparameters_with_specifier(best_params, unique_specifier, 'tune')
    
    logger.info(f"Hyperparameters saved for experiment: {unique_specifier}")


def run_train_mode(
    model_class,
    model_name: str,
    unique_specifier: str,
    data,
    dates,
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    args,
    train_tuned: bool
) -> None:
    """
    Run training mode with either tuned or default parameters.
    If train_tuned=True, uses K-fold cross validation with tuned parameters.
    If train_tuned=False, uses simple train/val split with default parameters.
    """
    logger = logging.getLogger(__name__)
    
    if train_tuned:
        # TUNED TRAINING: Use K-fold cross validation with tuned parameters
        logger.info(f"Starting tuned training with {args.k_folds}-fold cross validation")
        
        # Load tuned hyperparameters
        params = load_hyperparameters(unique_specifier, model_class, use_tuned=True)
        params['input_size'] = input_size
        params['sequence_length'] = args.sequence_length
        
        logger.info(f"Using tuned parameters: {params}")
        
        # Prepare data for k-fold cross validation
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
        fold_scores = []
        best_fold_score = float('inf')
        best_fold_model = None
        
        print(f"\nStarting {args.k_folds}-fold cross validation with tuned parameters...")
        
        # Convert data to numpy if it's a DataFrame
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = data
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(data_array)):
            print(f"\nFold {fold + 1}/{args.k_folds}")
            
            # Split data for this fold - use fold indices directly
            train_data = data_array[train_idx]
            val_data = data_array[val_idx]
            
            print(f"Fold {fold + 1} - Train: {len(train_data)}, Val: {len(val_data)}")
            
            # Create preprocessor for this fold
            from utils.data_preprocessing import TimeSeriesPreprocessor
            preprocessor = TimeSeriesPreprocessor(
                sequence_length=args.sequence_length,
                normalization=args.normalization if args.normalization != 'none' else None
            )
            
            # Fit scalers on training data only
            preprocessor.fit_scalers(train_data)
            
            # Create sequences for this fold
            X_train, y_train = preprocessor.create_sequences(train_data)
            X_val, y_val = preprocessor.create_sequences(val_data)
            
            print(f"Fold {fold + 1} sequences - Train: {X_train.shape}, Val: {X_val.shape}")
            
            # Create PyTorch datasets
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            # Create data loaders
            batch_size = 16
            fold_train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            fold_val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size
            )
            
            # Create model with tuned parameters
            model_params = filter_model_parameters(params)
            model = model_class(**model_params)
            trainer = TimeSeriesTrainer(model)
            
            # Train model
            history, metrics, predictions = trainer.train_and_evaluate(
                fold_train_loader,
                fold_val_loader,
                fold_val_loader,  # Use validation as test for cross-validation
                epochs=args.epochs,
                patience=args.patience,
                params=params
            )
            
            fold_score = metrics['val_loss']
            fold_scores.append(fold_score)
            
            print(f"Fold {fold + 1} validation loss: {fold_score:.4f}")
            
            # Keep track of best fold
            if fold_score < best_fold_score:
                best_fold_score = fold_score
                best_fold_model = model.state_dict()
            
            logger.info(f"Fold {fold + 1} completed with validation loss: {fold_score:.4f}")
        
        # Save best fold model weights
        if best_fold_model is not None:
            model_params = filter_model_parameters(params)
            final_model = model_class(**model_params)
            final_model.load_state_dict(best_fold_model)
            save_model_weights(final_model, unique_specifier, use_tuned=True)
        
        # Print cross-validation results
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\nK-Fold Cross Validation Results:")
        print(f"Mean validation loss: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print(f"Best fold validation loss: {best_fold_score:.4f}")
        
        logger.info(f"Tuned training completed. Mean loss: {mean_score:.4f}, Best loss: {best_fold_score:.4f}")
        
    else:
        # DEFAULT TRAINING: Use simple train/val split with default parameters
        logger.info(f"Starting default training with train/val split")
        
        # Get default parameters
        params = model_class.get_default_parameters()
        params['input_size'] = input_size
        params['sequence_length'] = args.sequence_length
        params['experiment_description'] = args.experiment_description
        
        logger.info(f"Using default parameters: {params}")
        
        # Create and train model
        model_params = filter_model_parameters(params)
        model = model_class(**model_params)
        trainer = TimeSeriesTrainer(model)
        
        print("\nTraining with default hyperparameters...")
        
        history, metrics, predictions = trainer.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            epochs=args.epochs,
            patience=args.patience,
            params=params
        )
        
        # Save default model weights
        save_model_weights(model, unique_specifier, use_tuned=False)
        
        # Save hyperparameters and results
        save_hyperparameters_with_specifier(params, unique_specifier, 'train')
        
        print(f"\nDefault Training Results:")
        print(f"Validation loss: {metrics['val_loss']:.4f}")
        print(f"Test loss: {metrics['test_loss']:.4f}")
        print(f"Test R² score: {metrics['test_r2']:.4f}")
        print(f"Test MAPE: {metrics['test_mape']:.2f}%")
        
        logger.info(f"Default training completed. Test loss: {metrics['test_loss']:.4f}")


def run_predict_mode(
    model_class,
    model_name: str,
    unique_specifier: str,
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    args,
    predict_tuned: bool
) -> None:
    """
    Run prediction mode using previously trained model.
    """
    logger = logging.getLogger(__name__)
    
    # Load appropriate hyperparameters
    params = load_hyperparameters(unique_specifier, model_class, use_tuned=predict_tuned)
    params['input_size'] = input_size
    params['sequence_length'] = args.sequence_length
    
    # Load model weights
    weights_path = load_model_weights(unique_specifier, use_tuned=predict_tuned)
    
    if not weights_path.exists():
        model_type = "tuned" if predict_tuned else "default"
        train_mode = "train --train_tuned true" if predict_tuned else "train --train_tuned false"
        raise FileNotFoundError(f"No {model_type} model weights found for {unique_specifier}. "
                              f"Please run '{train_mode}' mode first.")
    
    logger.info(f"Loading {'tuned' if predict_tuned else 'default'} model weights from: {weights_path}")
    
    # Create model and load weights
    model_params = filter_model_parameters(params)
    model = model_class(**model_params)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    trainer = TimeSeriesTrainer(model)
    
    print(f"\nRunning predictions with {'tuned' if predict_tuned else 'default'} model...")
    
    # Evaluate on test data
    criterion = torch.nn.MSELoss()
    test_loss, test_preds, test_targets, test_metrics = trainer.evaluate(test_loader, criterion)
    
    # Also evaluate on validation data for completeness
    val_loss, val_preds, val_targets, val_metrics = trainer.evaluate(val_loader, criterion)
    
    metrics = {
        'val_loss': val_loss,
        'val_r2': val_metrics['r2_score'],
        'val_mape': val_metrics['mape'],
        'test_loss': test_loss,
        'test_r2': test_metrics['r2_score'],
        'test_mape': test_metrics['mape']
    }
    
    predictions = {
        'val_predictions': val_preds,
        'val_targets': val_targets,
        'test_predictions': test_preds,
        'test_targets': test_targets
    }
    
    print(f"\nPrediction Results:")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test R² score: {test_metrics['r2_score']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")
    
    logger.info(f"Prediction completed. Test loss: {test_loss:.4f}")
    
    # Save prediction results
    save_results(model_name, {}, metrics, predictions, params, 
                mode='predict', experiment_description=args.experiment_description)


def run_report_mode(args) -> None:
    """
    Run report mode to show previous results.
    """
    print("\nReport Mode - Previous Results:")
    print("=" * 50)
    
    # This would need to be implemented based on your specific requirements
    # For now, just show a placeholder
    print("Report functionality to be implemented based on specific requirements.")
    print("This would show previous training/tuning results for comparison.")


def get_mode_description(mode: str) -> str:
    """
    Get description for each mode.
    
    Args:
        mode: Mode name
        
    Returns:
        Description string
    """
    descriptions = {
        'tune': 'Optimize hyperparameters using train/validation split and save tuned parameters',
        'train': 'Train model with either tuned (K-fold CV) or default (train/val split) hyperparameters',
        'predict': 'Load previously trained model and make predictions on test data',
        'report': 'Show previous training and tuning results'
    }
    return descriptions.get(mode, 'Unknown mode') 