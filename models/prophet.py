"""
TimeCast
Prophet model implementation for time series forecasting.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from models.base_model import BaseTimeSeriesModel
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Install with: pip install prophet")


class ProphetModel(BaseTimeSeriesModel):
    """
    Prophet model wrapper for time series forecasting.
    
    Note: Prophet works differently from deep learning models.
    It requires time series data in 'ds' (date) and 'y' (target) format.
    """
    
    def __init__(
        self,
        input_size: int,  # Keep for compatibility, not used in Prophet
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',  # 'additive' or 'multiplicative'
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000
    ):
        super().__init__()
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Please install with: pip install prophet")
        
        # Store parameters
        self.input_size = input_size  # Keep for compatibility
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        
        # Initialize Prophet model
        self.prophet_model = None
        self.fitted = False
        
        # Store data processing info
        self.data_scaler = None
        self.date_range = None
        
    def _create_prophet_model(self):
        """Create and configure Prophet model."""
        self.prophet_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples
        )
        
    def prepare_prophet_data(self, data: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Convert model input format to Prophet format.
        
        Prophet expects DataFrame with 'ds' (datestamp) and 'y' (target) columns.
        For merchant data, we sum across all merchants to get total consumption.
        """
        # Sum across merchants to get total consumption
        if len(data.shape) == 2:
            # Data is (timesteps, features) - sum across features
            y_values = np.sum(data, axis=1)
        else:
            # Data is already 1D
            y_values = data
            
        # Create date range if not provided
        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(y_values), freq='D')
        elif len(dates) != len(y_values):
            # Adjust dates to match data length
            dates = pd.date_range(start=dates[0], periods=len(y_values), freq='D')
            
        # Create Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': dates,
            'y': y_values
        })
        
        return prophet_df
        
    def fit(self, train_data: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """Fit Prophet model on training data."""
        # Create Prophet model
        self._create_prophet_model()
        
        # Prepare data for Prophet
        prophet_df = self.prepare_prophet_data(train_data, dates)
        
        # Fit model
        print("Fitting Prophet model...")
        self.prophet_model.fit(prophet_df)
        self.fitted = True
        
        return self
        
    def predict(self, n_periods: int = 1, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Make predictions using Prophet.
        
        Args:
            n_periods: Number of periods to forecast
            dates: Optional specific dates to predict for
            
        Returns:
            Array of predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if dates is not None:
            # Use provided dates
            future_df = pd.DataFrame({'ds': dates})
        else:
            # Create future dates
            future_df = self.prophet_model.make_future_dataframe(periods=n_periods, freq='D')
            
        # Make predictions
        forecast = self.prophet_model.predict(future_df)
        
        # Return predictions for the requested periods
        if dates is not None:
            return forecast['yhat'].values
        else:
            return forecast['yhat'].tail(n_periods).values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for compatibility with PyTorch training loop.
        Note: Prophet doesn't use this method directly.
        """
        # This is a placeholder for compatibility with the training framework
        # Prophet doesn't work with mini-batches like neural networks
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 1)
    
    def configure_optimizers(self):
        """Prophet doesn't use gradient-based optimization."""
        # Return a dummy optimizer for compatibility
        return torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=0.001)
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Return default parameters for Prophet."""
        return {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'mcmc_samples': 0,
            'interval_width': 0.80,
            'uncertainty_samples': 1000
        }
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges for hyperparameter tuning."""
        return {
            'changepoint_prior_scale': (0.001, 0.5),
            'seasonality_prior_scale': (0.01, 10),
            'holidays_prior_scale': (0.01, 10),
        }


class ProphetTrainer:
    """
    Custom trainer for Prophet model that works with the existing framework.
    """
    
    def __init__(self, model: ProphetModel):
        self.model = model
        
    def train_and_evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 1,  # Prophet doesn't use epochs
        patience: int = 10,  # Not used for Prophet
        params: Dict[str, Any] = None
    ) -> Tuple[Dict[str, list], Dict[str, float], Dict[str, Any]]:
        """
        Train Prophet model and evaluate performance.
        """
        # Convert DataLoader data to numpy arrays
        train_data, train_targets = self._extract_data_from_loader(train_loader)
        val_data, val_targets = self._extract_data_from_loader(val_loader)
        test_data, test_targets = self._extract_data_from_loader(test_loader)
        
        # Create date range for training data
        train_dates = pd.date_range(start='2023-01-01', periods=len(train_targets), freq='D')
        
        # Fit Prophet model
        self.model.fit(train_data, train_dates)
        
        # Make predictions
        val_predictions = self._predict_for_period(len(val_targets), len(train_targets))
        test_predictions = self._predict_for_period(len(test_targets), len(train_targets) + len(val_targets))
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        val_mse = mean_squared_error(val_targets, val_predictions)
        test_mse = mean_squared_error(test_targets, test_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        
        # Prepare return values
        history = {
            'train_loss': [0.0],  # Prophet doesn't have training loss
            'val_loss': [val_mse],
            'train_r2': [1.0],    # Placeholder
            'val_r2': [val_r2],
            'train_mape': [0.0],  # Calculate if needed
            'val_mape': [0.0]
        }
        
        metrics = {
            'val_loss': val_mse,
            'val_r2': val_r2,
            'val_mape': 0.0,  # Calculate if needed
            'test_loss': test_mse,
            'test_r2': test_r2,
            'test_mape': 0.0
        }
        
        predictions = {
            'val_predictions': val_predictions,
            'val_targets': val_targets,
            'test_predictions': test_predictions,
            'test_targets': test_targets
        }
        
        return history, metrics, predictions
    
    def _extract_data_from_loader(self, data_loader):
        """Extract data from PyTorch DataLoader."""
        all_data = []
        all_targets = []
        
        for batch_x, batch_y in data_loader:
            # For Prophet, we need to flatten the sequence dimension
            # and sum across features to get total consumption
            batch_data = batch_x.numpy()
            batch_targets = batch_y.numpy().flatten()
            
            # Sum across sequence and features to get total consumption per sample
            batch_consumption = np.sum(batch_data, axis=(1, 2))
            
            all_data.extend(batch_consumption)
            all_targets.extend(batch_targets)
        
        print(f"  Extracted {len(all_data)} samples from data loader")
        return np.array(all_data), np.array(all_targets)
    
    def _predict_for_period(self, n_periods: int, start_offset: int = 0):
        """Make predictions for a specific period."""
        if n_periods <= 0:
            print(f"  Warning: n_periods={n_periods}, returning empty array")
            return np.array([])
            
        # Create future dates
        base_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=start_offset)
        future_dates = pd.date_range(start=base_date, periods=n_periods, freq='D')
        
        print(f"  Predicting for {n_periods} periods starting from {base_date}")
        return self.model.predict(dates=future_dates) 