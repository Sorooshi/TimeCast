"""
Example: Merchant Data Preprocessing for TimeCast
================================================================

This script demonstrates how to preprocess raw merchant transaction data
(like merchant_synthetic.csv) to create the proper input format for our
time series forecasting models, following the mathematical formulation
in the LaTeX document.

LaTeX Formulation (Implemented):
- X_t ∈ ℝ^{N+contextual}: feature vector at time t (N merchants + contextual features)
- 𝒽_t ∈ ℝ^{(k+1)×(N+contextual)}: historical sequence matrix
- y_t = Σ x_{m,t}: total consumption across all merchants (sum of merchant columns)

Author: Soroosh Shalileh
Email: sr.shalileh@gmail.com
Year: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from utils.data_preprocessing import prepare_data_for_model
from utils.data_utils import load_and_validate_data


def load_merchant_transactions(file_path: str) -> pd.DataFrame:
    """
    Load raw merchant transaction data.
    
    Expected format:
    - timestamp: transaction timestamp
    - merchant_id: unique merchant identifier  
    - amount: transaction amount
    - other features: additional contextual features
    """
    print(f"📊 Loading raw transaction data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of unique merchants: {df['merchant_id'].nunique()}")
    print(f"Total transactions: {len(df)}")
    
    return df


def aggregate_merchant_data(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Aggregate transaction data to create merchant-level time series.
    
    This transforms raw transactions into the format expected by the LaTeX formulation:
    - Each row represents a time step t
    - Each column represents a merchant m
    - Values are x_{m,t} (total consumption for merchant m at time t)
    
    Args:
        df: Raw transaction data
        freq: Aggregation frequency ('D'=daily, 'H'=hourly, 'W'=weekly)
    
    Returns:
        DataFrame with DatetimeIndex and merchant columns
    """
    print(f"\n🔄 Aggregating data by {freq} frequency...")
    
    # Convert timestamp to datetime if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time-merchant pivot table
    # This creates X_t vectors where each row is time t and each column is merchant m
    merchant_data = df.pivot_table(
        index=pd.Grouper(key='timestamp', freq=freq),
        columns='merchant_id',
        values='amount',
        aggfunc='sum',
        fill_value=0.0
    )
    
    # Ensure column names are strings for consistency
    merchant_data.columns = [f'merchant_{int(col)}' for col in merchant_data.columns]
    
    print(f"Aggregated data shape: {merchant_data.shape}")
    print(f"Time steps (T): {len(merchant_data)}")
    print(f"Merchants (N): {len(merchant_data.columns)}")
    
    return merchant_data


def add_contextual_features(merchant_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add contextual features and target variable as per LaTeX formulation:
    - Time-of-day, day-of-week, holiday indicators, etc.
    - Target y_t = sum of merchant values at time t
    
    This expands from N merchants to N + contextual + target columns.
    """
    print(f"\n➕ Adding contextual features...")
    
    # Create a copy to avoid modifying original
    enhanced_data = merchant_data.copy()
    
    # Time-based features
    enhanced_data['hour'] = enhanced_data.index.hour
    enhanced_data['day_of_week'] = enhanced_data.index.dayofweek
    enhanced_data['is_weekend'] = enhanced_data.index.dayofweek.isin([5, 6]).astype(float)
    enhanced_data['month'] = enhanced_data.index.month
    enhanced_data['day_of_month'] = enhanced_data.index.day
    
    # Seasonal features
    enhanced_data['sin_month'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
    enhanced_data['cos_month'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
    enhanced_data['sin_hour'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['cos_hour'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
    
    # Simple holiday indicator (you could enhance this with actual holiday data)
    # For demonstration, mark some days as holidays
    holiday_dates = pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']).date
    enhanced_data['is_holiday'] = pd.Series(enhanced_data.index.date).isin(holiday_dates).astype(float).values
    
    # Add target column: y_t = sum of all merchant values at time t
    merchant_cols = [col for col in enhanced_data.columns if col.startswith('merchant_')]
    enhanced_data['total_consumption'] = enhanced_data[merchant_cols].sum(axis=1)
    
    print(f"Enhanced data shape: {enhanced_data.shape}")
    print(f"Added {enhanced_data.shape[1] - merchant_data.shape[1]} contextual features")
    print(f"Target column 'total_consumption' = sum of {len(merchant_cols)} merchant columns")
    
    return enhanced_data


def demonstrate_implementation_formulation(data: pd.DataFrame, sequence_length: int = 10):
    """
    Demonstrate how the preprocessed data maps to the LaTeX formulation.
    """
    print(f"\n📐 Demonstrating LaTeX Formulation Mapping:")
    print("=" * 50)
    
    merchant_cols = [col for col in data.columns if col.startswith('merchant_')]
    N = len(merchant_cols)  # Number of merchants
    total_features = data.shape[1]
    feature_count = total_features - 1  # Exclude target column
    T = len(data)
    k = sequence_length - 1  # k+1 = sequence_length, so k = sequence_length - 1
    
    print(f"LaTeX symbols → Implementation:")
    print(f"  N (merchants) = {N}")
    print(f"  Features (N + contextual) = {feature_count}")
    print(f"  Total columns = {total_features} (features + target)")
    print(f"  T (time steps) = {T}")  
    print(f"  k+1 (sequence length) = {sequence_length}")
    print(f"  k (lookback) = {k}")
    
    print(f"\nData structure:")
    print(f"  Raw data shape: {data.shape} → (T, N + contextual + target)")
    print(f"  After windowing: (samples, {sequence_length}, {feature_count})")
    print(f"  Target y_t = sum of merchant values (last column)")
    
    # Show example of X_t vector (merchants only)
    print(f"\nExample merchant values at time t=0:")
    example_merchants = data.iloc[0][merchant_cols].values
    print(f"  Merchants: {example_merchants}")
    print(f"  Sum: {example_merchants.sum():.2f}")
    
    # Show example target (should match sum)
    example_target = data.iloc[0][data.columns[-1]]  # Last column
    print(f"  y_0 (target): {example_target:.2f}")
    
    # Verify they match
    if abs(example_merchants.sum() - example_target) < 0.001:
        print(f"  ✅ Target matches sum of merchants")
    else:
        print(f"  ❌ Target mismatch!")


def visualize_data(data: pd.DataFrame, save_path: str = "merchant_data_analysis.png"):
    """Create visualizations to understand the data structure."""
    print(f"\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Total consumption over time
    merchant_cols = [col for col in data.columns if col.startswith('merchant_')]
    total_consumption = data[merchant_cols].sum(axis=1)
    
    axes[0, 0].plot(data.index, total_consumption)
    axes[0, 0].set_title('Total Consumption Over Time (y_t)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Total Amount')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Individual merchant patterns
    top_merchants = data[merchant_cols].sum().nlargest(5).index
    for merchant in top_merchants:
        axes[0, 1].plot(data.index, data[merchant], label=merchant, alpha=0.7)
    axes[0, 1].set_title('Top 5 Merchants (X_{m,t})')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Amount')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Correlation matrix of merchants
    merchant_corr = data[merchant_cols].corr()
    im = axes[1, 0].imshow(merchant_corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('Merchant Correlation Matrix')
    axes[1, 0].set_xlabel('Merchant ID')
    axes[1, 0].set_ylabel('Merchant ID')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Feature importance (variance)
    feature_vars = data.var().sort_values(ascending=False)
    axes[1, 1].bar(range(min(10, len(feature_vars))), feature_vars.head(10).values)
    axes[1, 1].set_title('Top 10 Features by Variance')
    axes[1, 1].set_xlabel('Feature Rank')
    axes[1, 1].set_ylabel('Variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def test_with_models(data: np.ndarray, sequence_length: int = 3):
    """
    Test the preprocessed data with our existing models to verify compatibility.
    """
    print(f"\n🧪 Testing compatibility with TimeSeriesPreprocessor...")
    
    # Adjust parameters for small dataset
    total_samples = len(data)
    print(f"Total samples: {total_samples}, Sequence length: {sequence_length}")
    
    # Calculate minimum samples needed per split
    min_samples_needed = sequence_length + 1
    print(f"Minimum samples needed per split: {min_samples_needed}")
    
    # Adjust split ratios to ensure each split has enough samples
    if total_samples < 20:
        # For very small datasets, use minimal splits
        train_ratio = 0.6
        val_ratio = 0.2  # This gives val=6 samples for 30 total
        print(f"Small dataset detected. Using train_ratio={train_ratio}, val_ratio={val_ratio}")
    else:
        train_ratio = 0.7
        val_ratio = 0.15
    
    try:
        # Use the existing data preparation pipeline (train mode returns 3 values)
        train_loader, val_loader, input_size = prepare_data_for_model(
            data=data,
            sequence_length=sequence_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=2,  # Very small batch size for small dataset
            mode='train'  # Specify mode to get consistent return values
        )
        
        print(f"✅ Data preprocessing successful!")
        print(f"  Input size: {input_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Check one batch
        for batch_x, batch_y in train_loader:
            print(f"  Batch shape: X={batch_x.shape}, y={batch_y.shape}")
            print(f"  Matches LaTeX: X=({batch_x.shape[0]}, {batch_x.shape[1]}, {batch_x.shape[2]}) = (batch, k+1, N)")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return False


def main():
    """
    Main function demonstrating the complete preprocessing pipeline.
    """
    print("🚀 Merchant Data Preprocessing Example")
    print("=" * 50)
    
    # Step 1: Load raw transaction data
    data_path = "data/merchant_synthetic.csv"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run the data generation script first or provide a valid path.")
        return
    
    df = load_merchant_transactions(data_path)
    
    # Step 2: Aggregate by time to create merchant time series
    merchant_data = aggregate_merchant_data(df, freq='D')  # Daily aggregation
    
    # Step 3: Add contextual features (optional)
    enhanced_data = add_contextual_features(merchant_data)
    
    # Step 4: Demonstrate LaTeX formulation mapping  
    sequence_length = 3  # Use small sequence length for demonstration dataset
    demonstrate_implementation_formulation(enhanced_data, sequence_length=sequence_length)
    
    # Step 5: Create visualizations
    visualize_data(enhanced_data)
    
    # Step 6: Test compatibility with existing models
    # Convert to numpy array for model testing
    data_array = enhanced_data.values
    success = test_with_models(data_array, sequence_length=sequence_length)
    
    # Step 7: Save preprocessed data
    output_path = "data/merchant_processed.csv"
    enhanced_data.to_csv(output_path)
    print(f"\n💾 Preprocessed data saved to: {output_path}")
    
    print(f"\n✅ Preprocessing pipeline completed successfully!")
    print(f"Data is ready for time series forecasting models.")
    
    return enhanced_data


if __name__ == "__main__":
    processed_data = main() 