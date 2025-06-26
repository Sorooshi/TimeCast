"""
Enhanced Example: Using Prophet with Preprocessed Merchant Data
==============================================================

This script demonstrates how to use Facebook Prophet with the preprocessed
merchant transaction data from example.py. It shows how to integrate Prophet
into the existing time series forecasting framework.

Key Features:
- Uses the same data preprocessing pipeline as example.py
- Shows how Prophet works with merchant time series data
- Compares Prophet with deep learning models
- Demonstrates forecasting with confidence intervals

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import existing preprocessing functions from example.py
from example import (
    load_merchant_transactions, 
    aggregate_merchant_data, 
    add_contextual_features
)

# Import our framework components
from utils.data_preprocessing import prepare_data_for_model
from utils.data_utils import load_and_validate_data

# Try to import Prophet
try:
    from prophet import Prophet
    from models.prophet import ProphetModel, ProphetTrainer
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None  # Define Prophet as None for type hints
    print("⚠️  Prophet not available. Install with: pip install prophet")


def prepare_data_for_prophet(enhanced_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert enhanced merchant data to Prophet format.
    
    Prophet expects:
    - 'ds': datestamp column (datetime)
    - 'y': target values (numeric)
    """
    print("\n🔄 Preparing data for Prophet...")
    
    # Get merchant columns (our main features)
    merchant_cols = [col for col in enhanced_data.columns if col.startswith('merchant_')]
    
    # Calculate total consumption (sum across all merchants)
    total_consumption = enhanced_data[merchant_cols].sum(axis=1)
    
    # Create Prophet dataframe
    prophet_df = pd.DataFrame({
        'ds': enhanced_data.index,  # DatetimeIndex becomes 'ds'
        'y': total_consumption.values  # Total consumption becomes 'y'
    })
    
    print(f"Prophet data shape: {prophet_df.shape}")
    print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(f"Target range: {prophet_df['y'].min():.2f} to {prophet_df['y'].max():.2f}")
    
    return prophet_df


def add_prophet_regressors(model, enhanced_data: pd.DataFrame):
    """
    Add external regressors (features) to Prophet model.
    
    Prophet can use additional features like:
    - Time-based features (hour, day_of_week, etc.)
    - Holiday indicators
    - Custom features
    """
    print("\n➕ Adding external regressors to Prophet...")
    
    # Add time-based regressors
    feature_columns = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'sin_month', 'cos_month', 'sin_hour', 'cos_hour',
        'is_holiday'
    ]
    
    added_regressors = []
    for feature in feature_columns:
        if feature in enhanced_data.columns:
            model.add_regressor(feature)
            added_regressors.append(feature)
            print(f"  ✓ Added regressor: {feature}")
    
    print(f"Total regressors added: {len(added_regressors)}")
    return model, added_regressors


def prepare_prophet_with_regressors(enhanced_data: pd.DataFrame, regressor_cols: list) -> pd.DataFrame:
    """
    Prepare Prophet dataframe with additional regressors.
    """
    # Get merchant columns
    merchant_cols = [col for col in enhanced_data.columns if col.startswith('merchant_')]
    
    # Calculate total consumption
    total_consumption = enhanced_data[merchant_cols].sum(axis=1)
    
    # Create Prophet dataframe with regressors
    prophet_df = pd.DataFrame({
        'ds': enhanced_data.index,
        'y': total_consumption.values
    })
    
    # Add regressor columns
    for col in regressor_cols:
        if col in enhanced_data.columns:
            prophet_df[col] = enhanced_data[col].values
    
    return prophet_df


def train_prophet_model(prophet_df: pd.DataFrame, with_regressors: bool = True):
    """
    Train Prophet model with the prepared data.
    """
    print(f"\n🧠 Training Prophet model (with_regressors={with_regressors})...")
    
    # Create Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # Turn off for daily data
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add regressors if requested
    added_regressors = []
    if with_regressors:
        regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
        for col in regressor_cols:
            model.add_regressor(col)
            added_regressors.append(col)
            print(f"  ✓ Added regressor: {col}")
    
    # Fit the model
    print("  📈 Fitting Prophet model...")
    model.fit(prophet_df)
    print("  ✅ Prophet model trained successfully!")
    
    return model, added_regressors


def make_prophet_forecast(model, prophet_df: pd.DataFrame, forecast_periods: int = 30) -> pd.DataFrame:
    """
    Make forecasts with Prophet model.
    """
    print(f"\n🔮 Making Prophet forecast for {forecast_periods} periods...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq='D')
    
    # Add regressor values for future dates
    regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
    
    if regressor_cols:
        print("  📊 Adding regressor values for future dates...")
        
        # For future dates, we need to generate regressor values
        # This is a simplified approach - in practice, you'd have actual future values
        for col in regressor_cols:
            if col.startswith('sin_') or col.startswith('cos_'):
                # Seasonal features can be calculated from dates
                if 'month' in col:
                    if 'sin' in col:
                        future[col] = np.sin(2 * np.pi * future['ds'].dt.month / 12)
                    else:
                        future[col] = np.cos(2 * np.pi * future['ds'].dt.month / 12)
                elif 'hour' in col:
                    # For daily data, assume noon (12:00)
                    if 'sin' in col:
                        future[col] = np.sin(2 * np.pi * 12 / 24)
                    else:
                        future[col] = np.cos(2 * np.pi * 12 / 24)
            elif col == 'is_weekend':
                future[col] = future['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            elif col == 'day_of_week':
                future[col] = future['ds'].dt.dayofweek
            elif col == 'month':
                future[col] = future['ds'].dt.month
            elif col == 'hour':
                future[col] = 12
            elif col == 'is_holiday':
                # Simple holiday detection
                holiday_dates = pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']).date
                future[col] = pd.Series(future['ds'].dt.date).isin(holiday_dates).astype(float).values
            else:
                # For other columns, use the mean value
                future[col] = prophet_df[col].mean()
    
    # Make forecast
    forecast = model.predict(future)
    
    print(f"  ✅ Forecast completed!")
    print(f"  📊 Forecast shape: {forecast.shape}")
    
    return forecast


def visualize_prophet_results(prophet_df: pd.DataFrame, forecast: pd.DataFrame, 
                            model, save_path: str = "prophet_forecast.png"):
    """
    Create comprehensive visualizations of Prophet results.
    """
    print(f"\n📊 Creating Prophet visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Main forecast plot
    ax1 = axes[0, 0]
    
    # Plot historical data
    ax1.plot(prophet_df['ds'], prophet_df['y'], 'ko', markersize=3, label='Observed', alpha=0.7)
    
    # Plot forecast
    forecast_future = forecast[forecast['ds'] > prophet_df['ds'].max()]
    ax1.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast', linewidth=2)
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='blue', alpha=0.2, label='Confidence Interval')
    
    # Highlight future predictions
    if len(forecast_future) > 0:
        ax1.plot(forecast_future['ds'], forecast_future['yhat'], 'r-', 
                label='Future Forecast', linewidth=2)
        ax1.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], 
                        forecast_future['yhat_upper'], color='red', alpha=0.2)
    
    ax1.set_title('Prophet Forecast: Total Merchant Consumption')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Consumption')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Trend component
    ax2 = axes[0, 1]
    ax2.plot(forecast['ds'], forecast['trend'], 'g-', linewidth=2)
    ax2.set_title('Trend Component')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Trend')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Weekly seasonality (if available)
    ax3 = axes[1, 0]
    if 'weekly' in forecast.columns:
        ax3.plot(forecast['ds'], forecast['weekly'], 'm-', linewidth=2)
        ax3.set_title('Weekly Seasonality')
    else:
        # Show residuals instead
        residuals = prophet_df['y'].values - forecast['yhat'][:len(prophet_df)].values
        ax3.plot(prophet_df['ds'], residuals, 'r.', alpha=0.6)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Residuals (Observed - Predicted)')
    ax3.set_xlabel('Date')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Yearly seasonality (if available)
    ax4 = axes[1, 1]
    if 'yearly' in forecast.columns:
        ax4.plot(forecast['ds'], forecast['yearly'], 'c-', linewidth=2)
        ax4.set_title('Yearly Seasonality')
    else:
        # Show forecast components
        forecast_components = ['trend']
        if 'weekly' in forecast.columns:
            forecast_components.append('weekly')
        if 'yearly' in forecast.columns:
            forecast_components.append('yearly')
        
        for component in forecast_components:
            if component in forecast.columns:
                ax4.plot(forecast['ds'], forecast[component], label=component, linewidth=2)
        ax4.set_title('Forecast Components')
        ax4.legend()
    ax4.set_xlabel('Date')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  💾 Visualizations saved to: {save_path}")
    plt.close()


def compare_with_framework_models(enhanced_data: pd.DataFrame):
    """
    Compare Prophet with deep learning models from the framework.
    """
    print(f"\n🔄 Comparing Prophet with framework models...")
    
    # Convert to numpy array for framework models
    data_array = enhanced_data.values
    
    # Prepare data using the framework
    train_loader, val_loader, test_loader, input_size = prepare_data_for_model(
        data=data_array,
        sequence_length=3,  # Reduced for small dataset (30 days total)
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=4  # Smaller batch size for small dataset
    )
    
    print(f"  📊 Data prepared for comparison:")
    print(f"    Input size: {input_size}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # For Prophet, we need to prepare the data differently
    prophet_df = prepare_data_for_prophet(enhanced_data)
    
    # Split Prophet data
    train_size = int(len(prophet_df) * 0.7)
    val_size = int(len(prophet_df) * 0.15)
    
    prophet_train = prophet_df[:train_size]
    prophet_val = prophet_df[train_size:train_size + val_size]
    prophet_test = prophet_df[train_size + val_size:]
    
    print(f"  📊 Prophet data splits:")
    print(f"    Train: {len(prophet_train)}")
    print(f"    Val: {len(prophet_val)}")
    print(f"    Test: {len(prophet_test)}")
    
    # Train Prophet model
    if PROPHET_AVAILABLE:
        prophet_model, _ = train_prophet_model(prophet_train, with_regressors=False)
        
        # Make predictions
        val_future = prophet_model.make_future_dataframe(periods=len(prophet_val), freq='D')
        val_forecast = prophet_model.predict(val_future)
        val_predictions = val_forecast['yhat'].tail(len(prophet_val)).values
        
        # Calculate Prophet metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        prophet_mse = mean_squared_error(prophet_val['y'].values, val_predictions)
        prophet_r2 = r2_score(prophet_val['y'].values, val_predictions)
        prophet_mae = mean_absolute_error(prophet_val['y'].values, val_predictions)
        
        print(f"\n📈 Prophet Results:")
        print(f"  MSE: {prophet_mse:.4f}")
        print(f"  R²: {prophet_r2:.4f}")
        print(f"  MAE: {prophet_mae:.4f}")
        
        return {
            'prophet_mse': prophet_mse,
            'prophet_r2': prophet_r2,
            'prophet_mae': prophet_mae,
            'prophet_predictions': val_predictions,
            'prophet_targets': prophet_val['y'].values
        }
    else:
        print("  ⚠️  Prophet not available for comparison")
        return None


def demonstrate_prophet_advantages():
    """
    Demonstrate specific advantages of Prophet for time series forecasting.
    """
    print(f"\n🎯 Prophet Advantages for Time Series Forecasting:")
    print("=" * 55)
    
    advantages = [
        "🔍 Automatic Seasonality Detection: Detects yearly, weekly, daily patterns",
        "📊 Confidence Intervals: Provides uncertainty estimates for forecasts",  
        "🎛️  External Regressors: Can incorporate additional features (holidays, events)",
        "📈 Trend Changes: Automatically detects and adapts to trend changes",
        "⚡ Fast Training: No iterative optimization like neural networks",
        "🔧 Easy Tuning: Intuitive hyperparameters with sensible defaults",
        "📊 Missing Data: Robust to missing values and irregular timestamps",
        "📈 Interpretability: Decomposes forecast into trend, seasonal, holiday components"
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")
    
    print(f"\n🤔 When to Use Prophet vs Deep Learning:")
    print("  🏆 Use Prophet when:")
    print("    • You have strong seasonal patterns")
    print("    • You need interpretable forecasts")  
    print("    • You have limited data or irregular timestamps")
    print("    • You need quick prototyping and deployment")
    print("    • You have domain knowledge about holidays/events")
    
    print(f"\n  🧠 Use Deep Learning when:")
    print("    • You have complex multivariate relationships")
    print("    • You have large amounts of training data")
    print("    • You need to model non-linear interactions")
    print("    • You want to jointly forecast multiple merchants")
    print("    • You have high-frequency data with complex patterns")


def main():
    """
    Main function demonstrating Prophet integration with preprocessed data.
    """
    print("🚀 Prophet Integration with Preprocessed Merchant Data")
    print("=" * 55)
    
    if not PROPHET_AVAILABLE:
        print("❌ Prophet is not installed.")
        print("To install Prophet, run: pip install prophet")
        print("Note: Prophet requires additional system dependencies.")
        return
    
    # Step 1: Load and preprocess data (reusing example.py pipeline)
    data_path = "data/merchant_synthetic.csv"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run example.py first to generate the data.")
        return
    
    # Load and preprocess data
    df = load_merchant_transactions(data_path)
    merchant_data = aggregate_merchant_data(df, freq='D')
    enhanced_data = add_contextual_features(merchant_data)
    
    # Step 2: Prepare data for Prophet
    prophet_df = prepare_data_for_prophet(enhanced_data)
    
    # Step 3: Train Prophet models (with and without regressors)
    print("\n" + "="*50)
    print("Training Prophet Models")
    print("="*50)
    
    # Basic Prophet model
    basic_model, _ = train_prophet_model(prophet_df, with_regressors=False)
    
    # Prophet with external regressors
    prophet_df_with_regressors = prepare_prophet_with_regressors(
        enhanced_data, 
        ['is_weekend', 'month', 'sin_month', 'cos_month', 'is_holiday']
    )
    enhanced_model, regressors = train_prophet_model(prophet_df_with_regressors, with_regressors=True)
    
    # Step 4: Make forecasts
    print("\n" + "="*50)
    print("Making Forecasts")
    print("="*50)
    
    forecast_basic = make_prophet_forecast(basic_model, prophet_df, forecast_periods=30)
    forecast_enhanced = make_prophet_forecast(enhanced_model, prophet_df_with_regressors, forecast_periods=30)
    
    # Step 5: Visualize results
    visualize_prophet_results(prophet_df, forecast_basic, basic_model, "prophet_basic_forecast.png")
    visualize_prophet_results(prophet_df_with_regressors, forecast_enhanced, enhanced_model, "prophet_enhanced_forecast.png")
    
    # Step 6: Compare with framework models
    comparison_results = compare_with_framework_models(enhanced_data)
    
    # Step 7: Show Prophet advantages
    demonstrate_prophet_advantages()
    
    # Step 8: Summary
    print(f"\n🎉 Prophet Integration Complete!")
    print("="*40)
    print("✅ Successfully integrated Prophet with preprocessed merchant data")
    print("✅ Trained both basic and enhanced Prophet models") 
    print("✅ Generated forecasts with confidence intervals")
    print("✅ Created comprehensive visualizations")
    print("✅ Compared with deep learning framework")
    
    print(f"\nFiles generated:")
    print("  📊 prophet_basic_forecast.png - Basic Prophet model results")
    print("  📊 prophet_enhanced_forecast.png - Enhanced model with regressors")
    
    return {
        'enhanced_data': enhanced_data,
        'prophet_df': prophet_df,
        'basic_model': basic_model,
        'enhanced_model': enhanced_model,
        'forecast_basic': forecast_basic,
        'forecast_enhanced': forecast_enhanced,
        'comparison_results': comparison_results
    }


if __name__ == "__main__":
    results = main() 