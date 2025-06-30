"""
TimeCast
Training utilities and hyperparameter tuning functionality.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
import optuna
from sklearn.metrics import r2_score
from models.base_model import BaseTimeSeriesModel

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

class TimeSeriesTrainer:
    def __init__(
        self,
        model: BaseTimeSeriesModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output = self.model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> float:
        """Simple validation that only returns loss (during training)"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        preprocessor=None
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluate the model on the given data loader.
        
        Returns:
            loss: Average loss value
            predictions: Model predictions
            targets: True target values
            metrics: Dictionary containing R² and MAPE scores
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                
                predictions.append(output.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Denormalize for meaningful metrics if preprocessor available
        if preprocessor is not None:
            predictions_denorm = preprocessor.denormalize_targets(predictions)
            targets_denorm = preprocessor.denormalize_targets(targets)
            
            # Calculate metrics on denormalized values
            r2 = r2_score(targets_denorm, predictions_denorm)
            mape = calculate_mape(targets_denorm, predictions_denorm)
        else:
            # Fallback to normalized scale
            r2 = r2_score(targets, predictions)
            mape = calculate_mape(targets, predictions)
        
        metrics = {
            'r2_score': r2,
            'mape': mape
        }
        
        return total_loss / len(data_loader), predictions, targets, metrics

    def train_and_evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 10,
        params: Dict[str, Any] = None
    ) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, Any]]:
        """
        Train the model and evaluate on validation and test sets.
        
        Returns:
            history: Training history (losses and metrics)
            metrics: Best validation and test metrics (loss, R², MAPE)
            predictions: Predictions on validation and test sets
        """
        optimizer = self.model.configure_optimizers()
        criterion = torch.nn.MSELoss()
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_r2': [],
            'train_mape': [],
            'val_loss': [],
            'val_r2': [],
            'val_mape': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            _, train_preds, train_targets, train_metrics = self.evaluate(train_loader, criterion)
               
            # Validate
            val_loss, val_preds, val_targets, val_metrics = self.evaluate(val_loader, criterion)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_r2'].append(train_metrics['r2_score'])
            history['train_mape'].append(train_metrics['mape'])
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_metrics['r2_score'])
            history['val_mape'].append(val_metrics['mape'])
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation on validation and test sets
        val_loss, val_preds, val_targets, val_metrics = self.evaluate(val_loader, criterion)
        test_loss, test_preds, test_targets, test_metrics = self.evaluate(test_loader, criterion)
        
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
        
        return history, metrics, predictions


def tune_hyperparameters(
    model_class,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_trials: int = 100,
    epochs: int = 100,
    patience: int = 25,
    input_size: int = None,
    sequence_length: int = 10
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune hyperparameters using Optuna.
    
    Args:
        model_class: Model class to tune
        train_loader: Training data loader
        val_loader: Validation data loader
        n_trials: Number of trials for hyperparameter search
        epochs: Maximum number of epochs per trial
        patience: Early stopping patience
        input_size: Number of input features
        sequence_length: Length of input sequences
        sequence_length: Length of input sequences
        
    Returns:
        best_params: Best hyperparameters found
        metrics: Validation metrics for best model
    """
    def objective(trial):
        # Create a temporary model instance to get parameter ranges
        # Only pass sequence_length to models that need it (like MLP)
        if model_class.__name__ == 'MLP':
            temp_model = model_class(input_size=input_size, sequence_length=sequence_length)
            params = {'input_size': input_size, 'sequence_length': sequence_length}
        else:
            temp_model = model_class(input_size=input_size)
            params = {'input_size': input_size}
        
        param_ranges = temp_model.get_parameter_ranges()
        hidden_sizes = []  # Track hidden sizes separately
        
        for param_name, range_value in param_ranges.items():
            if param_name == 'hidden_sizes':
                # Handle hidden_sizes parameter specially
                for i, (low, high) in enumerate(range_value):
                    size = trial.suggest_int(f'hidden_size_{i}', low, high)
                    hidden_sizes.append(size)
            elif isinstance(range_value, tuple):
                low, high = range_value
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
        
        # Add hidden_sizes to params if any were collected
        if hidden_sizes:
            params['hidden_sizes'] = hidden_sizes

        # Initialize model with suggested parameters
        model = model_class(**params)
        trainer = TimeSeriesTrainer(model)
        
        try:
            # Train and evaluate
            history, metrics, _ = trainer.train_and_evaluate(
                train_loader,
                val_loader,
                val_loader,  # Use validation set for both validation and test
                epochs=epochs,
                patience=patience,
                params=params
            )
            return metrics['val_loss']
        except RuntimeError as e:
            print(f"Trial failed with error: {str(e)}")
            print(f"Parameters used: {params}")
            raise e
    
    # Create and run study with logging
    study = optuna.create_study(direction='minimize')
    
    # Set up logging for optuna if available
    import logging
    logger = logging.getLogger(__name__)
    
    # Add callback to log trial results
    def logging_callback(study, trial):
        logger.info(f"Trial {trial.number} finished with value: {trial.value:.4f}")
        logger.info(f"Trial {trial.number} parameters: {trial.params}")
    
    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])
    
    # Get best parameters and reconstruct hidden_sizes if needed
    if model_class.__name__ == 'MLP':
        best_params = {'input_size': input_size, 'sequence_length': sequence_length}
    else:
        best_params = {'input_size': input_size}
    hidden_sizes = []
    
    # Extract parameters from study's best trial
    for param_name, param_value in study.best_params.items():
        if param_name.startswith('hidden_size_'):
            hidden_sizes.append(param_value)
        else:
            best_params[param_name] = param_value
            
    # Add hidden_sizes to best_params if any were collected
    if hidden_sizes:
        best_params['hidden_sizes'] = hidden_sizes
    
    # Train final model with best parameters to get metrics
    model = model_class(**best_params)
    trainer = TimeSeriesTrainer(model)
    _, metrics, _ = trainer.train_and_evaluate(
        train_loader,
        val_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        params=best_params
    )
    
    return best_params, metrics 