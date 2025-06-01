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
    - transaction_speed: processing time in seconds
    - customer_loyalty_score: customer loyalty score (0-100)
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
    
    # Add transaction_speed (processing time in seconds)
    # Base speed of 2-5 seconds, with slower processing during peak hours and holidays
    base_speed = np.random.uniform(2, 5, len(df))
    peak_hours = (df['hour'] >= 12) & (df['hour'] <= 14)  # Lunch hours
    speed_multipliers = (
        (1.5 * peak_hours.astype(float)) +  # Slower during peak hours
        (1.3 * df['is_holiday'].astype(float)) +  # Slower during holidays
        (1.2 * df['is_weekend'].astype(float)) +  # Slightly slower on weekends
        1.0  # Base multiplier
    )
    df['transaction_speed'] = base_speed * speed_multipliers
    
    # Add customer_loyalty_score (0-100)
    # Base scores with some randomness
    base_scores = np.random.normal(60, 15, n_customers)
    base_scores = np.clip(base_scores, 0, 100)
    
    # Create a mapping from customer_id to loyalty score
    loyalty_scores = {i+1: score for i, score in enumerate(base_scores)}
    
    # Add scores to DataFrame
    df['customer_loyalty_score'] = df['customer_id'].map(loyalty_scores)
    
    # Round numeric columns
    df['amount'] = np.round(df['amount'], 2)
    df['transaction_speed'] = np.round(df['transaction_speed'], 2)
    df['customer_loyalty_score'] = np.round(df['customer_loyalty_score'], 1)
    
    # Reorder columns
    columns = ['timestamp', 'merchant_id', 'customer_id', 'amount', 
              'day_of_week', 'hour', 'is_weekend', 'is_holiday',
              'transaction_speed', 'customer_loyalty_score']
    
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