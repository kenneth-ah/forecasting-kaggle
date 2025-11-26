import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def improved_forecast(train_path, output_path, forecast_weeks=None):
    """
    Improved forecasting model with trend, lags, and rolling statistics.
    Combines seasonal patterns with trend and recent history.
    
    Args:
        train_path: Path to training data CSV
        output_path: Path where the submission CSV will be saved
        forecast_weeks: List of weeks to forecast
    """
    
    # Read training data
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Week'])
    
    # Define forecast weeks for July-August 2025
    if forecast_weeks is None:
        forecast_weeks = [
            '2025-07-06', '2025-07-13', '2025-07-20', '2025-07-27',
            '2025-08-03', '2025-08-10', '2025-08-17', '2025-08-24', '2025-08-31'
        ]
    
    forecast_weeks = pd.to_datetime(forecast_weeks)
    
    submission_rows = []
    
    for forecast_date in forecast_weeks:
        # Date one year back (seasonal)
        previous_year_date = forecast_date - pd.DateOffset(years=1)
        # Date one week back (trend)
        previous_week_date = forecast_date - pd.DateOffset(weeks=1)
        
        items = df['item_id'].unique()
        
        for item_id in items:
            item_data = df[df['item_id'] == item_id].copy()
            item_data = item_data.sort_values('date')
            
            # Feature 1: Seasonal component (same week last year)
            seasonal_data = item_data[item_data['date'] == previous_year_date]
            seasonal_value = seasonal_data['search_volume'].values[0] if len(seasonal_data) > 0 else None
            
            # Feature 2: Recent trend (last 4 weeks average)
            recent_data = item_data[
                (item_data['date'] >= forecast_date - pd.DateOffset(weeks=5)) &
                (item_data['date'] < forecast_date)
            ]
            recent_avg = recent_data['search_volume'].mean() if len(recent_data) > 0 else None
            
            # Feature 3: Trend direction (comparing 4 weeks ago to 8 weeks ago)
            early_data = item_data[
                (item_data['date'] >= forecast_date - pd.DateOffset(weeks=12)) &
                (item_data['date'] < forecast_date - pd.DateOffset(weeks=8))
            ]
            early_avg = early_data['search_volume'].mean() if len(early_data) > 0 else None
            
            # Feature 4: Item volatility (std dev of all historical data)
            volatility = item_data['search_volume'].std()
            is_stable = volatility < item_data['search_volume'].std() / 2  # Adjust based on item stability
            
            # Feature 5: Overall item average
            item_avg = item_data['search_volume'].mean()
            
            # Combine features with weighted ensemble
            forecast_value = None
            
            # If we have seasonal data, use it as base
            if seasonal_value is not None:
                forecast_value = seasonal_value
                
                # Adjust for trend if available
                if recent_avg is not None and early_avg is not None:
                    trend = recent_avg - early_avg
                    # Apply trend adjustment more conservatively for volatile items
                    trend_weight = 0.2 if not is_stable else 0.3
                    forecast_value = forecast_value + (trend * trend_weight)
            
            # Fallback to recent average if no seasonal data
            elif recent_avg is not None:
                forecast_value = recent_avg
            
            # Final fallback to item average
            else:
                forecast_value = item_avg
            
            # Ensure non-negative and reasonable bounds
            forecast_value = max(0, min(100, int(forecast_value)))
            
            submission_rows.append({
                'id': f"{forecast_date.strftime('%Y-%m-%d')}{int(item_id)}",
                'Week': forecast_date.strftime('%d-%m-%Y'),
                'item_id': int(item_id),
                'search_volume_forecast': forecast_value
            })
    
    # Create and save submission
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Improved Forecast Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Forecast weeks: {len(forecast_weeks)}")
    print(f"Number of items: {len(submission_df['item_id'].unique())}")
    print(f"\nFirst few rows:\n{submission_df.head(10)}")
    print(f"\nStatistics of forecasts:")
    print(f"Mean: {submission_df['search_volume_forecast'].mean():.2f}")
    print(f"Median: {submission_df['search_volume_forecast'].median():.2f}")
    print(f"Min: {submission_df['search_volume_forecast'].min()}")
    print(f"Max: {submission_df['search_volume_forecast'].max()}")
    print(f"Std Dev: {submission_df['search_volume_forecast'].std():.2f}")

if __name__ == "__main__":
    train_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "train.csv"
    )
    output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "submissions",
        "improved_submission.csv"
    )
    
    improved_forecast(train_file, output_file)
