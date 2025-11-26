import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

def create_time_features(df):
    """Create time-based features from date column"""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1

    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    # NEW: Add December indicator
    df['is_december'] = (df['month'] == 12).astype(int)

    # NEW: Add summer indicator (July or August)
    df['is_summer'] = df['month'].isin([7, 8]).astype(int)

    return df

def create_lag_features(df, item_id, lags=[1, 2, 3, 4, 8, 12, 52]):
    """Create lag features for a specific item"""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # Rolling statistics
    for window in [4, 8, 12, 26, 52]:
        df[f'rolling_mean_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).min()

    return df

def apply_holtwinters(series, seasonal_periods=52, forecast_steps=9):
    """
    Apply Holt-Winters Exponential Smoothing

    Args:
        series: Time series data
        seasonal_periods: Number of periods in a season (52 for weekly data with yearly seasonality)
        forecast_steps: Number of steps to forecast

    Returns:
        forecast values and fitted values
    """
    try:
        # Ensure we have enough data
        if len(series) < 2 * seasonal_periods:
            # Not enough data for seasonal model, use simple exponential smoothing
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
        else:
            # Use full Holt-Winters with seasonality
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )

        fitted_model = model.fit(optimized=True)

        # Get fitted values for training data
        fitted_values = fitted_model.fittedvalues

        # Forecast future values
        forecast = fitted_model.forecast(steps=forecast_steps)

        return forecast, fitted_values
    except Exception as e:
        print(f"Holt-Winters failed: {e}. Using simple mean forecast.")
        mean_val = series.mean()
        return np.array([mean_val] * forecast_steps), np.array([mean_val] * len(series))

def holtwinters_lightgbm_forecast(train_path, output_path, forecast_weeks=None):
    """
    Combined Holt-Winters and LightGBM forecasting model.

    Strategy:
    1. Apply Holt-Winters to get baseline forecasts and capture seasonality
    2. Use Holt-Winters forecasts and residuals as features for LightGBM
    3. Train LightGBM with time-series features and Holt-Winters features
    4. Generate final predictions

    Args:
        train_path: Path to training data CSV
        output_path: Path where the submission CSV will be saved
        forecast_weeks: List of weeks to forecast (if None, uses standard competition weeks)
    """

    print("Loading training data...")
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
    n_forecast_weeks = len(forecast_weeks)

    print(f"Forecasting for {n_forecast_weeks} weeks from {forecast_weeks[0]} to {forecast_weeks[-1]}")

    # Get all items
    items = df['item_id'].unique()
    print(f"Number of items: {len(items)}")

    # Store all predictions
    all_predictions = []

    # Process each item separately
    for idx, item_id in enumerate(items):
        if (idx + 1) % 20 == 0:
            print(f"Processing item {idx + 1}/{len(items)}")

        # Filter data for this item
        item_data = df[df['item_id'] == item_id].copy()
        item_data = item_data.sort_values('date').reset_index(drop=True)

        # Get category information
        main_category = item_data['Maincategory'].iloc[0]
        sub_category = item_data['Subcategory'].iloc[0]

        # Step 1: Apply Holt-Winters
        series = item_data['search_volume'].values
        hw_forecast, hw_fitted = apply_holtwinters(series, seasonal_periods=52, forecast_steps=n_forecast_weeks)

        # Add Holt-Winters fitted values to training data
        item_data['hw_fitted'] = hw_fitted
        item_data['hw_residual'] = item_data['search_volume'] - item_data['hw_fitted']

        # Step 2: Create features for LightGBM
        item_data = create_time_features(item_data)
        item_data = create_lag_features(item_data, item_id)

        # Create category encoding (simple label encoding)
        item_data['main_cat_encoded'] = pd.Categorical(item_data['Maincategory']).codes
        item_data['sub_cat_encoded'] = pd.Categorical(item_data['Subcategory']).codes

        # Step 3: Prepare training data for LightGBM
        # Remove rows with NaN (due to lag features)
        train_data = item_data.dropna()

        if len(train_data) < 50:
            # Not enough data for LightGBM, use Holt-Winters forecast only
            predictions = hw_forecast
        else:
            # Define features for LightGBM
            feature_cols = [col for col in train_data.columns if col not in
                          ['id', 'Week', 'item_id', 'Maincategory', 'Subcategory',
                           'search_volume', 'date']]

            X_train = train_data[feature_cols]
            y_train = train_data['search_volume']

            # Train LightGBM model
            lgb_params = {
                'objective': 'regression',
                'metric': 'mape',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'min_data_in_leaf': 10,
                'max_depth': 6
            }

            train_set = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(lgb_params, train_set, num_boost_round=100, valid_sets=[train_set],
                            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

            # Step 4: Create future features for prediction
            last_date = item_data['date'].max()
            future_dates = []
            predictions = []

            # For each forecast week
            for i, forecast_date in enumerate(forecast_weeks):
                # Create a row for this future date
                future_row = pd.DataFrame({
                    'date': [forecast_date],
                    'item_id': [item_id],
                    'Maincategory': [main_category],
                    'Subcategory': [sub_category]
                })

                # Add time features
                future_row = create_time_features(future_row)
                future_row['main_cat_encoded'] = pd.Categorical([main_category],
                                                                categories=df['Maincategory'].unique()).codes[0]
                future_row['sub_cat_encoded'] = pd.Categorical([sub_category],
                                                              categories=df['Subcategory'].unique()).codes[0]

                # Add Holt-Winters forecast as feature
                future_row['hw_fitted'] = hw_forecast[i]
                future_row['hw_residual'] = 0  # Unknown for future

                # Add lag features by extending the series
                extended_series = np.append(series, predictions)  # Use previous predictions

                for lag in [1, 2, 3, 4, 8, 12, 52]:
                    lag_idx = len(extended_series) - lag
                    if lag_idx >= 0:
                        future_row[f'lag_{lag}'] = extended_series[lag_idx]
                    else:
                        future_row[f'lag_{lag}'] = series.mean()

                # Add rolling features
                for window in [4, 8, 12, 26, 52]:
                    window_data = extended_series[-window:] if len(extended_series) >= window else extended_series
                    future_row[f'rolling_mean_{window}'] = np.mean(window_data) if len(window_data) > 0 else series.mean()
                    future_row[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0
                    future_row[f'rolling_max_{window}'] = np.max(window_data) if len(window_data) > 0 else series.max()
                    future_row[f'rolling_min_{window}'] = np.min(window_data) if len(window_data) > 0 else series.min()

                # Ensure all feature columns are present
                for col in feature_cols:
                    if col not in future_row.columns:
                        future_row[col] = 0

                # Make prediction
                X_future = future_row[feature_cols]
                pred = model.predict(X_future, num_iteration=model.best_iteration)[0]
                pred = max(0, pred)  # Ensure non-negative
                predictions.append(pred)

        # Store predictions for this item
        for i, forecast_date in enumerate(forecast_weeks):
            all_predictions.append({
                'id': f"{forecast_date.strftime('%Y-%m-%d')}{int(item_id)}",
                'Week': forecast_date.strftime('%d-%m-%Y'),
                'item_id': int(item_id),
                'search_volume_forecast': int(round(predictions[i]))
            })

    # Create submission dataframe
    print("\nCreating submission file...")
    submission_df = pd.DataFrame(all_predictions)

    # Sort by id for consistency
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    # Save submission
    submission_df.to_csv(output_path, index=False)

    print(f"\nHolt-Winters + LightGBM Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Forecast weeks: {len(forecast_weeks)}")
    print(f"Number of items: {len(submission_df['item_id'].unique())}")
    print(f"\nFirst few rows:\n{submission_df.head(10)}")
    print(f"\nStatistics of forecasts:")
    print(f"Mean: {submission_df['search_volume_forecast'].mean():.2f}")
    print(f"Median: {submission_df['search_volume_forecast'].median():.2f}")
    print(f"Min: {submission_df['search_volume_forecast'].min()}")
    print(f"Max: {submission_df['search_volume_forecast'].max()}")

    return submission_df

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
        "holtwinters_lightgbm_submission.csv"
    )

    holtwinters_lightgbm_forecast(train_file, output_file)

