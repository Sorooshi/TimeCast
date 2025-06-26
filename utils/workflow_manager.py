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
import os
import json
import pandas as pd
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
    print(f"\n📊 Report Mode - {args.report_type.upper()}")
    print("=" * 70)
    
    # Base paths
    base_path = Path(".").resolve()
    hyperparams_path = base_path / "Hyperparameters"
    weights_path = base_path / "Weights"
    results_path = base_path / "Results"
    
    if args.report_type in ['all', 'models']:
        show_available_models(hyperparams_path, weights_path)
    
    if args.report_type in ['all', 'performance']:
        show_performance_comparison(hyperparams_path, results_path)
    
    if args.report_type in ['all', 'best']:
        show_best_configurations(hyperparams_path, results_path)
    
    if args.report_type in ['all', 'timeline']:
        show_experiment_timeline(hyperparams_path, weights_path)
    
    if args.report_type in ['all', 'files']:
        show_file_paths(hyperparams_path, weights_path, results_path)


def show_available_models(hyperparams_path: Path, weights_path: Path) -> None:
    """Show available trained models."""
    print("\n🤖 AVAILABLE TRAINED MODELS")
    print("-" * 70)
    
    if not hyperparams_path.exists() or not weights_path.exists():
        print("❌ No trained models found. Run training first.")
        return
    
    # Collect model information
    models_info = {}
    
    # Parse hyperparameter files
    if hyperparams_path.exists():
        for hp_file in hyperparams_path.glob("*.json"):
            try:
                parts = hp_file.stem.split('_')
                if len(parts) >= 4:
                    model_name = parts[0]
                    data_name = parts[1]
                    exp_desc = '_'.join(parts[2:-2])
                    seq_len = parts[-2]
                    mode_type = parts[-1]  # 'tuned' or 'train'
                    
                    key = f"{model_name}_{data_name}_{exp_desc}_{seq_len}"
                    if key not in models_info:
                        models_info[key] = {
                            'model': model_name,
                            'data': data_name,
                            'experiment': exp_desc,
                            'sequence_length': seq_len,
                            'has_tuned': False,
                            'has_default': False,
                            'tuned_weights': False,
                            'default_weights': False
                        }
                    
                    if mode_type == 'tuned':
                        models_info[key]['has_tuned'] = True
                    elif mode_type == 'train':
                        models_info[key]['has_default'] = True
            except Exception as e:
                continue
    
    # Check for weight files
    if weights_path.exists():
        for weight_file in weights_path.glob("*.pth"):
            try:
                stem = weight_file.stem
                if stem.endswith('_tuned_best'):
                    key = stem.replace('_tuned_best', '')
                    if key in models_info:
                        models_info[key]['tuned_weights'] = True
                elif stem.endswith('_default_best'):
                    key = stem.replace('_default_best', '')
                    if key in models_info:
                        models_info[key]['default_weights'] = True
            except Exception as e:
                continue
    
    if not models_info:
        print("❌ No trained models found.")
        return
    
    # Display models in a table format
    print(f"{'Model':<12} {'Data':<20} {'Experiment':<25} {'Seq':<4} {'Tuned':<6} {'Default':<7} {'Status':<10}")
    print("-" * 70)
    
    for key, info in sorted(models_info.items()):
        tuned_status = "✅" if info['has_tuned'] and info['tuned_weights'] else "❌"
        default_status = "✅" if info['has_default'] and info['default_weights'] else "❌"
        
        # Overall status
        if info['tuned_weights'] and info['default_weights']:
            status = "Complete"
        elif info['tuned_weights'] or info['default_weights']:
            status = "Partial"
        else:
            status = "No Weights"
        
        print(f"{info['model']:<12} {info['data']:<20} {info['experiment']:<25} "
              f"{info['sequence_length']:<4} {tuned_status:<6} {default_status:<7} {status:<10}")
    
    print(f"\n📈 Total models: {len(models_info)}")
    complete_models = sum(1 for info in models_info.values() 
                         if info['tuned_weights'] and info['default_weights'])
    print(f"🎯 Complete models (both tuned & default): {complete_models}")


def show_performance_comparison(hyperparams_path: Path, results_path: Path) -> None:
    """Show performance comparison tables."""
    print("\n📊 PERFORMANCE COMPARISON")
    print("-" * 70)
    
    # Collect performance data from hyperparameter files and results
    performance_data = []
    
    # Look for results in Results directory
    if results_path.exists():
        for model_dir in results_path.iterdir():
            if model_dir.is_dir():
                for mode_dir in model_dir.iterdir():
                    if mode_dir.is_dir():
                        for exp_dir in mode_dir.iterdir():
                            if exp_dir.is_dir():
                                summary_file = exp_dir / "summary.json"
                                if summary_file.exists():
                                    try:
                                        with open(summary_file, 'r') as f:
                                            summary = json.load(f)
                                        
                                        metrics = summary.get('metrics', {})
                                        hyperparams = summary.get('hyperparameters', {})
                                        
                                        performance_data.append({
                                            'model': model_dir.name,
                                            'mode': mode_dir.name,
                                            'experiment': summary.get('experiment_description', 'N/A'),
                                            'sequence_length': hyperparams.get('sequence_length', 'N/A'),
                                            'val_loss': float(metrics.get('val_loss', 0)),
                                            'test_loss': float(metrics.get('test_loss', 0)),
                                            'val_r2': float(metrics.get('val_r2', 0)),
                                            'test_r2': float(metrics.get('test_r2', 0)),
                                            'val_mape': float(metrics.get('val_mape', 0)),
                                            'test_mape': float(metrics.get('test_mape', 0))
                                        })
                                    except Exception as e:
                                        continue
    
    if not performance_data:
        print("❌ No performance data found. Run experiments first.")
        return
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(performance_data)
    
    # Group by model and show comparison
    print("\n🏆 BEST PERFORMANCE BY MODEL:")
    print(f"{'Model':<12} {'Best Test Loss':<15} {'Best Test R²':<12} {'Best Test MAPE':<12} {'Experiment':<25}")
    print("-" * 70)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        best_loss = model_data.loc[model_data['test_loss'].idxmin()]
        best_r2 = model_data.loc[model_data['test_r2'].idxmax()]
        best_mape = model_data.loc[model_data['test_mape'].idxmin()]
        
        print(f"{model:<12} {best_loss['test_loss']:<15.4f} {best_r2['test_r2']:<12.4f} "
              f"{best_mape['test_mape']:<12.2f} {best_loss['experiment']:<25}")
    
    # Show detailed comparison table
    print(f"\n📋 DETAILED PERFORMANCE TABLE:")
    print(f"{'Model':<8} {'Mode':<6} {'Experiment':<20} {'Seq':<4} {'Test Loss':<10} {'Test R²':<8} {'Test MAPE':<10}")
    print("-" * 70)
    
    # Sort by test loss (best first)
    df_sorted = df.sort_values('test_loss')
    
    for _, row in df_sorted.head(20).iterrows():  # Show top 20
        print(f"{row['model']:<8} {row['mode']:<6} {str(row['experiment'])[:20]:<20} "
              f"{row['sequence_length']:<4} {row['test_loss']:<10.4f} "
              f"{row['test_r2']:<8.4f} {row['test_mape']:<10.2f}")
    
    if len(df_sorted) > 20:
        print(f"... and {len(df_sorted) - 20} more results")


def show_best_configurations(hyperparams_path: Path, results_path: Path) -> None:
    """Show best performing configurations."""
    print("\n🏆 BEST PERFORMING CONFIGURATIONS")
    print("-" * 70)
    
    # Collect all results with hyperparameters
    configs_data = []
    
    if results_path.exists():
        for model_dir in results_path.iterdir():
            if model_dir.is_dir():
                for mode_dir in model_dir.iterdir():
                    if mode_dir.is_dir():
                        for exp_dir in mode_dir.iterdir():
                            if exp_dir.is_dir():
                                summary_file = exp_dir / "summary.json"
                                if summary_file.exists():
                                    try:
                                        with open(summary_file, 'r') as f:
                                            summary = json.load(f)
                                        
                                        metrics = summary.get('metrics', {})
                                        hyperparams = summary.get('hyperparameters', {})
                                        
                                        config = {
                                            'model': model_dir.name,
                                            'mode': mode_dir.name,
                                            'experiment': summary.get('experiment_description', 'N/A'),
                                            'test_loss': float(metrics.get('test_loss', float('inf'))),
                                            'test_r2': float(metrics.get('test_r2', 0)),
                                            'hyperparams': hyperparams
                                        }
                                        configs_data.append(config)
                                    except Exception as e:
                                        continue
    
    if not configs_data:
        print("❌ No configuration data found.")
        return
    
    # Find best configurations
    configs_data.sort(key=lambda x: x['test_loss'])
    
    print("🥇 TOP 5 BEST CONFIGURATIONS BY TEST LOSS:")
    print("-" * 70)
    
    for i, config in enumerate(configs_data[:5], 1):
        print(f"\n{i}. {config['model']} - {config['experiment']}")
        print(f"   Test Loss: {config['test_loss']:.4f} | Test R²: {config['test_r2']:.4f}")
        print(f"   Key Hyperparameters:")
        
        # Show important hyperparameters
        important_params = ['learning_rate', 'batch_size', 'hidden_size', 'num_layers', 
                          'dropout', 'sequence_length', 'epochs']
        
        for param in important_params:
            if param in config['hyperparams']:
                print(f"     {param}: {config['hyperparams'][param]}")


def show_experiment_timeline(hyperparams_path: Path, weights_path: Path) -> None:
    """Show experiment timeline based on file modification times."""
    print("\n⏰ EXPERIMENT TIMELINE")
    print("-" * 70)
    
    timeline_data = []
    
    # Collect file modification times
    paths_to_check = [hyperparams_path, weights_path]
    
    for path in paths_to_check:
        if path.exists():
            for file_path in path.glob("*.json" if "Hyperparameters" in str(path) else "*.pth"):
                try:
                    mtime = file_path.stat().st_mtime
                    timestamp = datetime.fromtimestamp(mtime)
                    
                    # Parse filename
                    parts = file_path.stem.split('_')
                    if len(parts) >= 4:
                        model_name = parts[0]
                        data_name = parts[1]
                        exp_desc = '_'.join(parts[2:-2])
                        
                        file_type = "Hyperparameters" if file_path.suffix == ".json" else "Weights"
                        
                        timeline_data.append({
                            'timestamp': timestamp,
                            'model': model_name,
                            'data': data_name,
                            'experiment': exp_desc,
                            'type': file_type,
                            'file': file_path.name
                        })
                except Exception as e:
                    continue
    
    if not timeline_data:
        print("❌ No experiment files found.")
        return
    
    # Sort by timestamp (newest first)
    timeline_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    print(f"{'Date':<19} {'Time':<8} {'Model':<8} {'Data':<15} {'Type':<15} {'Experiment':<20}")
    print("-" * 70)
    
    for entry in timeline_data[:30]:  # Show last 30 experiments
        date_str = entry['timestamp'].strftime('%Y-%m-%d')
        time_str = entry['timestamp'].strftime('%H:%M:%S')
        print(f"{date_str:<19} {time_str:<8} {entry['model']:<8} {entry['data']:<15} "
              f"{entry['type']:<15} {entry['experiment']:<20}")
    
    if len(timeline_data) > 30:
        print(f"... and {len(timeline_data) - 30} older experiments")
    
    # Show summary statistics
    print(f"\n📅 Timeline Summary:")
    print(f"   Total experiments: {len(timeline_data)}")
    if timeline_data:
        oldest = min(timeline_data, key=lambda x: x['timestamp'])
        newest = max(timeline_data, key=lambda x: x['timestamp'])
        print(f"   Date range: {oldest['timestamp'].strftime('%Y-%m-%d')} to {newest['timestamp'].strftime('%Y-%m-%d')}")


def show_file_paths(hyperparams_path: Path, weights_path: Path, results_path: Path) -> None:
    """Show file paths for weights and hyperparameters."""
    print("\n📁 FILE PATHS")
    print("-" * 70)
    
    print("📂 Directory Structure:")
    print(f"   Hyperparameters: {hyperparams_path}")
    print(f"   Weights: {weights_path}")
    print(f"   Results: {results_path}")
    
    # Show hyperparameter files
    print(f"\n📄 HYPERPARAMETER FILES:")
    if hyperparams_path.exists():
        hp_files = list(hyperparams_path.glob("*.json"))
        hp_files.sort()
        
        if hp_files:
            print(f"{'Filename':<50} {'Size':<10} {'Modified':<20}")
            print("-" * 70)
            
            for file_path in hp_files:
                try:
                    size = file_path.stat().st_size
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
                    
                    print(f"{file_path.name:<50} {size_str:<10} {mtime.strftime('%Y-%m-%d %H:%M'):<20}")
                except Exception as e:
                    print(f"{file_path.name:<50} {'Error':<10} {'N/A':<20}")
        else:
            print("   ❌ No hyperparameter files found")
    else:
        print("   ❌ Hyperparameters directory not found")
    
    # Show weight files
    print(f"\n⚖️  WEIGHT FILES:")
    if weights_path.exists():
        weight_files = list(weights_path.glob("*.pth"))
        weight_files.sort()
        
        if weight_files:
            print(f"{'Filename':<50} {'Size':<10} {'Modified':<20}")
            print("-" * 70)
            
            for file_path in weight_files:
                try:
                    size = file_path.stat().st_size
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024*1024:
                        size_str = f"{size//1024}KB"
                    else:
                        size_str = f"{size//(1024*1024)}MB"
                    
                    print(f"{file_path.name:<50} {size_str:<10} {mtime.strftime('%Y-%m-%d %H:%M'):<20}")
                except Exception as e:
                    print(f"{file_path.name:<50} {'Error':<10} {'N/A':<20}")
        else:
            print("   ❌ No weight files found")
    else:
        print("   ❌ Weights directory not found")
    
    # Show results files
    print(f"\n📊 RESULTS FILES:")
    if results_path.exists():
        result_files = []
        for root, dirs, files in os.walk(results_path):
            for file in files:
                if file.endswith('.json'):
                    result_files.append(Path(root) / file)
        
        if result_files:
            result_files.sort()
            print(f"{'Relative Path':<60} {'Size':<10} {'Modified':<20}")
            print("-" * 70)
            
            for file_path in result_files:
                try:
                    size = file_path.stat().st_size
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
                    
                    rel_path = file_path.relative_to(results_path)
                    print(f"{str(rel_path):<60} {size_str:<10} {mtime.strftime('%Y-%m-%d %H:%M'):<20}")
                except Exception as e:
                    print(f"{str(file_path):<60} {'Error':<10} {'N/A':<20}")
        else:
            print("   ❌ No results files found")
    else:
        print("   ❌ Results directory not found")
    
    print(f"\n💾 Storage Summary:")
    total_size = 0
    file_count = 0
    
    for path in [hyperparams_path, weights_path]:
        if path.exists():
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except:
                        pass
    
    if total_size < 1024*1024:
        size_str = f"{total_size//1024}KB"
    else:
        size_str = f"{total_size//(1024*1024)}MB"
    
    print(f"   Total files: {file_count}")
    print(f"   Total size: {size_str}")


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