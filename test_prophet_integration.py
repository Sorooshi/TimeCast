"""
Test Script: Prophet Integration
===============================

Simple test to verify Prophet works with the preprocessing pipeline.

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_basic_prophet_integration():
    """Test basic Prophet integration with synthetic data."""
    print("🧪 Testing Prophet Integration")
    print("=" * 40)
    
    # Check if Prophet is available
    try:
        from prophet import Prophet
        print("✅ Prophet is available")
    except ImportError:
        print("❌ Prophet not installed. Install with: pip install prophet")
        return False
    
    # Create synthetic time series data
    print("\n📊 Creating synthetic time series data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create synthetic merchant data with seasonal patterns
    np.random.seed(42)
    n_merchants = 5
    n_days = len(dates)
    
    # Base trend
    trend = np.linspace(100, 200, n_days)
    
    # Add weekly seasonality (higher on weekends)
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Add yearly seasonality 
    yearly_pattern = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Create merchant data
    merchant_data = []
    for i in range(n_merchants):
        merchant_values = (
            trend + 
            weekly_pattern + 
            yearly_pattern + 
            10 * np.random.randn(n_days) +  # noise
            20 * i  # merchant-specific offset
        )
        merchant_data.append(np.maximum(merchant_values, 0))  # ensure positive
    
    # Create DataFrame
    merchant_df = pd.DataFrame(
        {f'merchant_{i}': merchant_data[i] for i in range(n_merchants)},
        index=dates
    )
    
    print(f"  Synthetic data shape: {merchant_df.shape}")
    print(f"  Date range: {merchant_df.index.min()} to {merchant_df.index.max()}")
    
    # Calculate total consumption
    total_consumption = merchant_df.sum(axis=1)
    
    # Prepare Prophet data
    prophet_df = pd.DataFrame({
        'ds': merchant_df.index,
        'y': total_consumption.values
    })
    
    print(f"  Prophet data shape: {prophet_df.shape}")
    print(f"  Total consumption range: {prophet_df['y'].min():.2f} to {prophet_df['y'].max():.2f}")
    
    # Test Prophet training
    print("\n🧠 Training Prophet model...")
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        model.fit(prophet_df)
        print("  ✅ Prophet model trained successfully!")
        
        # Test forecasting
        print("\n🔮 Testing forecasting...")
        future = model.make_future_dataframe(periods=30)  # 30 days ahead
        forecast = model.predict(future)
        
        print(f"  ✅ Forecast generated successfully!")
        print(f"  Forecast shape: {forecast.shape}")
        print(f"  Future predictions (next 5 days):")
        
        future_forecasts = forecast.tail(30)
        for i, (_, row) in enumerate(future_forecasts.head(5).iterrows()):
            print(f"    Day +{i+1}: {row['yhat']:.2f} (±{row['yhat_upper'] - row['yhat']:.2f})")
        
        # Test components
        print("\n📊 Testing component decomposition...")
        components = ['trend', 'weekly', 'yearly']
        available_components = [comp for comp in components if comp in forecast.columns]
        print(f"  Available components: {available_components}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error during Prophet training: {str(e)}")
        return False


def test_prophet_with_preprocessing():
    """Test Prophet with the existing preprocessing pipeline."""
    print("\n🔄 Testing Prophet with Preprocessing Pipeline")
    print("=" * 50)
    
    try:
        from utils.data_preprocessing import prepare_data_for_model
        print("✅ Preprocessing module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing preprocessing: {str(e)}")
        return False
    
    # Create synthetic data in the format expected by the preprocessing pipeline
    np.random.seed(123)
    n_timesteps = 100
    n_features = 3  # 3 merchants
    
    # Create synthetic multivariate time series
    data = np.random.randn(n_timesteps, n_features) * 10 + 50
    data = np.maximum(data, 0)  # ensure positive values
    
    print(f"  Created synthetic data: {data.shape}")
    print(f"  Value range: {data.min():.2f} to {data.max():.2f}")
    
    # Test preprocessing
    try:
        train_loader, val_loader, test_loader, input_size = prepare_data_for_model(
            data=data,
            sequence_length=5,
            train_ratio=0.7,
            val_ratio=0.15,
            batch_size=4
        )
        
        print(f"  ✅ Preprocessing successful!")
        print(f"    Input size: {input_size}")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        print(f"    Test batches: {len(test_loader)}")
        
        # Extract data for Prophet format
        print("\n🔄 Converting to Prophet format...")
        
        # For Prophet, we need to flatten and sum the data
        total_consumption = np.sum(data, axis=1)  # sum across merchants
        dates = pd.date_range(start='2023-01-01', periods=len(total_consumption), freq='D')
        
        prophet_df = pd.DataFrame({
            'ds': dates,
            'y': total_consumption
        })
        
        print(f"  ✅ Prophet format conversion successful!")
        print(f"    Prophet data shape: {prophet_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in preprocessing pipeline: {str(e)}")
        return False


def main():
    """Run all Prophet integration tests."""
    print("🚀 Prophet Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Basic Prophet functionality
    test1_passed = test_basic_prophet_integration()
    
    # Test 2: Prophet with preprocessing pipeline
    test2_passed = test_prophet_with_preprocessing()
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    print(f"Basic Prophet Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Prophet with Preprocessing: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Prophet is ready to use.")
        print("\nNext steps:")
        print("1. Install Prophet: pip install prophet")
        print("2. Run example_with_prophet.py for comprehensive demo")
        print("3. Use main.py --model Prophet for framework integration")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("Make sure Prophet is properly installed with all dependencies.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 