import torch
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import KFold
import pandas as pd
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

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module
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
        
        # Calculate additional metrics
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
            history: Training history (losses)
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
            'val_loss': [],
            'val_r2': [],
            'val_mape': []
        }
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_preds, val_targets, val_metrics = self.evaluate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
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

    def k_fold_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        batch_size: int = 32,
        epochs: int = 100,
        params: Dict[str, Any] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size
            )

            optimizer = self.model.configure_optimizers()
            criterion = torch.nn.MSELoss()

            best_val_loss = float('inf')
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer, criterion)
                val_loss = self.validate(val_loader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            fold_scores.append(best_val_loss)

        return fold_scores, params

def tune_hyperparameters(
    model_class: type,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_trials: int = 100,
    epochs: int = 100,
    patience: int = 10
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Tune hyperparameters using Optuna.
    
    Returns:
        best_params: Best hyperparameters found
        best_metrics: Best validation metrics achieved
    """
    def objective(trial: optuna.Trial) -> float:
        # Get parameter ranges from model
        model = model_class()
        param_ranges = model.get_parameter_ranges()
        
        # Create parameters dictionary from ranges
        params = {}
        for param_name, (low, high) in param_ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)

        # Initialize model with suggested parameters
        model = model_class(**params)
        trainer = TimeSeriesTrainer(model)
        
        # Train and evaluate
        history, metrics, _ = trainer.train_and_evaluate(
            train_loader,
            val_loader,
            val_loader,  # Using val_loader as dummy test_loader during tuning
            epochs=epochs,
            patience=patience
        )
        
        # Report additional metrics
        trial.set_user_attr('r2_score', metrics['val_r2'])
        trial.set_user_attr('mape', metrics['val_mape'])
        
        return metrics['val_loss']

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_metrics = {
        'val_loss': study.best_value,
        'val_r2': study.best_trial.user_attrs['r2_score'],
        'val_mape': study.best_trial.user_attrs['mape']
    }

    return study.best_params, best_metrics 