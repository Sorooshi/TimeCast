import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_merchant_data(n_days=30, n_merchants=5, n_customers=50):
    """
    Generate synthetic merchant transaction data with the following features:
    - timestamp: datetime of the transaction
    - merchant_id: identifier for the merchant
    - customer_id: identifier for the customer
    - amount: transaction amount
    - day_of_week: day of the week (0-6)
    - hour: hour of the day (0-23)
    - is_weekend: boolean flag for weekend
    - is_holiday: boolean flag for holidays (randomly assigned)
    """
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    dates = []
    for day in range(n_days):
        # Generate more transactions during business hours
        n_transactions = np.random.poisson(50)  # Average 50 transactions per day (reduced from 100)
        for _ in range(n_transactions):
            # More transactions during business hours (8am-6pm)
            hour = np.random.normal(14, 4)  # Center around 2 PM
            hour = int(np.clip(hour, 0, 23))
            minute = np.random.randint(0, 60)
            dates.append(start_date + timedelta(days=day, hours=hour, minutes=minute))
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': sorted(dates),
        'merchant_id': np.random.randint(1, n_merchants + 1, len(dates)),
        'customer_id': np.random.randint(1, n_customers + 1, len(dates))
    })
    
    # Add amount with seasonal and time-based patterns
    base_amount = np.random.lognormal(3, 1, len(df))  # Base transaction amount
    
    # Add seasonal patterns
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['month'] = df['timestamp'].dt.month
    
    # Adjust amounts based on patterns
    amount_multipliers = (
        (1.2 * df['is_weekend'].astype(float)) +  # Weekend uplift
        (0.3 * np.sin(2 * np.pi * df['month'] / 12)) +  # Monthly seasonality
        (0.2 * np.sin(2 * np.pi * df['hour'] / 24)) +  # Daily seasonality
        1.0  # Base multiplier
    )
    
    df['amount'] = base_amount * amount_multipliers
    
    # Add random holidays
    n_holidays = n_days // 20  # Approximately 18 holidays per year
    holiday_dates = pd.date_range(start=start_date, periods=n_days).to_pydatetime()
    holiday_dates = np.random.choice(holiday_dates, n_holidays, replace=False)
    df['is_holiday'] = df['timestamp'].dt.date.isin(holiday_dates)
    
    # Increase amounts during holidays
    df.loc[df['is_holiday'], 'amount'] *= 1.5
    
    # Round amounts to 2 decimal places
    df['amount'] = np.round(df['amount'], 2)
    
    # Reorder columns
    columns = ['timestamp', 'merchant_id', 'customer_id', 'amount', 
              'day_of_week', 'hour', 'is_weekend', 'is_holiday']
    
    return df[columns]

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    print("Generating synthetic merchant transaction data...")
    df = generate_merchant_data()
    
    # Save to CSV
    output_file = 'data/merchant_synthetic.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(f"Shape of the dataset: {df.shape}")
    print("\nSample of the data:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Additional information
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

if __name__ == "__main__":
    main() 