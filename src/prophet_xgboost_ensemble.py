import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
import xgboost as xgb
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

    # Is summer month indicator (July/August)
    df['is_summer'] = df['month'].isin([7, 8]).astype(int)

    return df

def create_advanced_lag_features(df, lags=[1, 2, 4, 8, 12, 26, 52]):
    """Create advanced lag and rolling features"""
    df = df.copy()

    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # Rolling statistics - multiple windows
    for window in [4, 8, 12, 26, 52]:
        df[f'rolling_mean_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).min()
        df[f'rolling_median_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).median()

    # Exponential weighted moving averages
    for span in [4, 12, 26]:
        df[f'ewm_{span}'] = df['search_volume'].shift(1).ewm(span=span).mean()

    # Momentum and trend features
    df['momentum_4'] = df['search_volume'].shift(1) - df['search_volume'].shift(5)
    df['momentum_12'] = df['search_volume'].shift(1) - df['search_volume'].shift(13)

    # Year-over-year change
    df['yoy_change'] = df['search_volume'].shift(1) - df['search_volume'].shift(53)

    return df

def apply_prophet(item_data, forecast_steps=9):
    """
    Apply Facebook Prophet for forecasting

    Args:
        item_data: DataFrame with 'date' and 'search_volume' columns
        forecast_steps: Number of steps to forecast

    Returns:
        forecast values and fitted values
    """
    try:
        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            'ds': item_data['date'],
            'y': item_data['search_volume']
        })

        # Initialize Prophet with tuned parameters for Google Trends data
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # We have weekly data, so weekly seasonality doesn't make sense
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Google Trends often has multiplicative seasonality
            changepoint_prior_scale=0.05,  # Moderate flexibility for trend changes
            seasonality_prior_scale=10.0,  # Strong seasonality
            interval_width=0.95,
            uncertainty_samples=0  # Faster training
        )

        # Add custom seasonalities
        model.add_seasonality(name='quarterly', period=91.25/7, fourier_order=5)  # Quarterly in weeks

        # Fit the model
        model.fit(prophet_df)

        # Get fitted values
        fitted = model.predict(prophet_df)
        fitted_values = fitted['yhat'].values

        # Create future dataframe for forecasting
        last_date = item_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                      periods=forecast_steps, freq='W')
        future_df = pd.DataFrame({'ds': future_dates})

        # Make forecast
        forecast = model.predict(future_df)
        forecast_values = forecast['yhat'].values

        # Ensure non-negative forecasts (Google Trends can't be negative)
        forecast_values = np.maximum(forecast_values, 0)
        fitted_values = np.maximum(fitted_values, 0)

        return forecast_values, fitted_values

    except Exception as e:
        print(f"Prophet failed: {e}. Using simple mean forecast.")
        mean_val = item_data['search_volume'].mean()
        return np.array([mean_val] * forecast_steps), np.array([mean_val] * len(item_data))

def create_category_aggregates(df):
    """Create category-level aggregates as additional features"""
    category_features = df.groupby(['date', 'Subcategory'])['search_volume'].agg([
        ('cat_mean', 'mean'),
        ('cat_std', 'std'),
        ('cat_median', 'median')
    ]).reset_index()

    df = df.merge(category_features, on=['date', 'Subcategory'], how='left')

    return df

def prophet_xgboost_ensemble(train_path, output_path, forecast_weeks=None):
    """
    Prophet + XGBoost Ensemble forecasting model with hierarchical features.

    Strategy:
    1. Apply Prophet to capture seasonality and trend
    2. Create hierarchical features from category aggregates
    3. Train XGBoost with Prophet forecasts + advanced features
    4. Ensemble Prophet and XGBoost predictions

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

    # Create category-level aggregates
    print("Creating category-level features...")
    df = create_category_aggregates(df)

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

        # Step 1: Apply Prophet
        prophet_forecast, prophet_fitted = apply_prophet(item_data, forecast_steps=n_forecast_weeks)

        # Add Prophet fitted values to training data
        item_data['prophet_fitted'] = prophet_fitted
        item_data['prophet_residual'] = item_data['search_volume'] - item_data['prophet_fitted']

        # Step 2: Create features for XGBoost
        item_data = create_time_features(item_data)
        item_data = create_advanced_lag_features(item_data)

        # Create category encoding
        item_data['main_cat_encoded'] = pd.Categorical(item_data['Maincategory']).codes
        item_data['sub_cat_encoded'] = pd.Categorical(item_data['Subcategory']).codes

        # Add ratio to category mean
        item_data['ratio_to_cat_mean'] = item_data['search_volume'] / (item_data['cat_mean'] + 1)

        # Step 3: Prepare training data for XGBoost
        # Remove rows with NaN (due to lag features)
        train_data = item_data.dropna()

        if len(train_data) < 50:
            # Not enough data for XGBoost, use Prophet forecast only
            predictions = prophet_forecast
        else:
            # Define features for XGBoost
            feature_cols = [col for col in train_data.columns if col not in
                          ['id', 'Week', 'item_id', 'Maincategory', 'Subcategory',
                           'search_volume', 'date']]

            X_train = train_data[feature_cols]
            y_train = train_data['search_volume']

            # Train XGBoost model with optimized parameters
            xgb_params = {
                'objective': 'reg:absoluteerror',  # MAE objective aligns with MAPE
                'eval_metric': 'mape',
                'learning_rate': 0.03,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

            model = xgb.XGBRegressor(**xgb_params)

            # Use last 20% of data as validation for early stopping
            split_point = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_point]
            y_train_split = y_train.iloc[:split_point]
            X_val = X_train.iloc[split_point:]
            y_val = y_train.iloc[split_point:]

            if len(X_val) > 0:
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)

            # Step 4: Create forecast features
            # We need to build features iteratively for each forecast step
            forecast_data = []
            current_data = item_data.copy()

            for i, forecast_date in enumerate(forecast_weeks):
                # Create a row for this forecast date
                forecast_row = pd.DataFrame({
                    'date': [forecast_date],
                    'Maincategory': [main_category],
                    'Subcategory': [sub_category],
                })

                # Add time features
                forecast_row = create_time_features(forecast_row)

                # Add prophet forecast for this step
                forecast_row['prophet_fitted'] = prophet_forecast[i]
                forecast_row['prophet_residual'] = 0  # Unknown, set to 0

                # Add category features (use last known values or prophet-based estimates)
                last_cat_mean = current_data['cat_mean'].iloc[-1]
                forecast_row['cat_mean'] = last_cat_mean
                forecast_row['cat_std'] = current_data['cat_std'].iloc[-1]
                forecast_row['cat_median'] = current_data['cat_median'].iloc[-1]
                forecast_row['ratio_to_cat_mean'] = prophet_forecast[i] / (last_cat_mean + 1)

                # Add category encoding
                forecast_row['main_cat_encoded'] = current_data['main_cat_encoded'].iloc[0]
                forecast_row['sub_cat_encoded'] = current_data['sub_cat_encoded'].iloc[0]

                # Combine with historical data to create lag features
                temp_data = pd.concat([current_data, forecast_row], ignore_index=True)

                # For the first iteration, we don't have search_volume yet, use prophet forecast
                if i == 0:
                    temp_data.loc[temp_data.index[-1], 'search_volume'] = prophet_forecast[i]

                temp_data = create_advanced_lag_features(temp_data)

                # Get features for this forecast point
                forecast_features = temp_data.iloc[-1:][feature_cols]

                # Make XGBoost prediction
                xgb_pred = model.predict(forecast_features)[0]
                xgb_pred = max(xgb_pred, 0)  # Ensure non-negative

                # Ensemble: weighted average of Prophet and XGBoost
                # Give more weight to Prophet for items with less training data
                # and more weight to XGBoost for items with more stable patterns

                data_quality_weight = min(len(train_data) / 200, 1.0)  # 0 to 1 based on data length
                cv = train_data['search_volume'].std() / (train_data['search_volume'].mean() + 1)
                stability_weight = max(1 - cv, 0.3)  # Higher weight for more stable series

                xgb_weight = 0.5 * data_quality_weight * stability_weight
                prophet_weight = 1 - xgb_weight

                final_pred = prophet_weight * prophet_forecast[i] + xgb_weight * xgb_pred
                final_pred = max(final_pred, 0)  # Ensure non-negative

                forecast_data.append(final_pred)

                # Update current_data with the prediction for next iteration
                forecast_row['search_volume'] = final_pred
                current_data = pd.concat([current_data, forecast_row], ignore_index=True)

            predictions = np.array(forecast_data)

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        # Store predictions for this item
        for i, forecast_date in enumerate(forecast_weeks):
            week_str = forecast_date.strftime('%Y-%m-%d')
            submission_id = f"{week_str}{item_id}"

            all_predictions.append({
                'id': submission_id,
                'search_volume': predictions[i]
            })

    # Create submission dataframe
    print("\nCreating submission file...")
    submission_df = pd.DataFrame(all_predictions)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")

    return submission_df

if __name__ == "__main__":
    train_path = "../data/train.csv"
    output_path = "../submissions/prophet_xgboost_submission.csv"

    # Run the ensemble forecast
    submission = prophet_xgboost_ensemble(train_path, output_path)

    print("\nSample predictions:")
    print(submission.head(20))

