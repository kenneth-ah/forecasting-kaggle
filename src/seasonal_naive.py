import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def seasonal_naive_forecast(train_path, output_path, forecast_weeks=None):
    """
    Seasonal Naive forecasting model.
    Uses the search volume from the same week in the previous year as the forecast.
    
    Args:
        train_path: Path to training data CSV
        output_path: Path where the submission CSV will be saved
        forecast_weeks: List of weeks to forecast (if None, uses standard competition weeks)
    """
    
    # Read training data
    df = pd.read_csv(train_path)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['Week'])
    
    # Define forecast weeks for July-August 2025
    if forecast_weeks is None:
        forecast_weeks = [
            '2025-07-06',
            '2025-07-13',
            '2025-07-20',
            '2025-07-27',
            '2025-08-03',
            '2025-08-10',
            '2025-08-17',
            '2025-08-24',
            '2025-08-31'
        ]
    
    forecast_weeks = pd.to_datetime(forecast_weeks)
    
    # Create submission dataframe
    submission_rows = []
    
    for forecast_date in forecast_weeks:
        # Calculate the corresponding date from previous year
        previous_year_date = forecast_date - pd.DateOffset(years=1)
        
        # Get all items
        items = df['item_id'].unique()
        
        for item_id in items:
            # Filter data for this item and the previous year date
            item_data = df[(df['item_id'] == item_id) & (df['date'] == previous_year_date)]
            
            if len(item_data) > 0:
                # Use the search volume from previous year
                forecast_value = item_data['search_volume'].values[0]
            else:
                # If no data for exact date, use the average of that item
                item_avg = df[df['item_id'] == item_id]['search_volume'].mean()
                forecast_value = max(0, int(item_avg))  # Use item average, at least 0
            
            submission_rows.append({
                'id': f"{forecast_date.strftime('%Y-%m-%d')}{int(item_id)}",
                'Week': forecast_date.strftime('%d-%m-%Y'),
                'item_id': int(item_id),
                'search_volume_forecast': int(forecast_value)
            })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(submission_rows)
    
    # Sort by id for consistency
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    
    print(f"Seasonal Naive Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Forecast weeks: {len(forecast_weeks)}")
    print(f"Number of items: {len(submission_df['item_id'].unique())}")
    print(f"\nFirst few rows:\n{submission_df.head(10)}")
    print(f"\nStatistics of forecasts:")
    print(f"Mean: {submission_df['search_volume_forecast'].mean():.2f}")
    print(f"Median: {submission_df['search_volume_forecast'].median():.2f}")
    print(f"Min: {submission_df['search_volume_forecast'].min()}")
    print(f"Max: {submission_df['search_volume_forecast'].max()}")

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
        "seasonal_naive_submission.csv"
    )
    
    seasonal_naive_forecast(train_file, output_file)
