"""
SARIMA + Gradient Boosting Hybrid Model

This approach combines traditional statistical methods (SARIMA) with modern ML (Gradient Boosting).
It's faster than Prophet and often more accurate for Google Trends data.

Key advantages:
1. SARIMA explicitly models seasonality, trend, and AR components
2. Uses SARIMA residuals as features for GB model
3. Faster training than Prophet
4. Works well with Google Trends weekly patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

def fit_sarima_model(series, order=(1,1,1), seasonal_order=(1,1,1,52), forecast_steps=9):
    """
    Fit SARIMA model with seasonal period of 52 weeks

    Args:
        series: Time series data
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) seasonal order
        forecast_steps: Number of steps to forecast

    Returns:
        forecast, fitted values
    """
    try:
        # Handle series with insufficient data
        if len(series) < 104:  # Less than 2 years
            seasonal_order = (0, 0, 0, 0)  # No seasonal component

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(disp=False, maxiter=50)

        # Get fitted values
        fitted_values = fitted_model.fittedvalues

        # Forecast
        forecast = fitted_model.forecast(steps=forecast_steps)

        return np.array(forecast), np.array(fitted_values)

    except Exception as e:
        print(f"SARIMA failed: {e}. Using mean.")
        mean_val = series.mean()
        return np.array([mean_val] * forecast_steps), np.array([mean_val] * len(series))

def create_features(df):
    """Create time series features"""
    df = df.copy()

    # Time features
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Summer indicator
    df['is_summer'] = df['month'].isin([7, 8]).astype(int)

    # Lags
    for lag in [1, 2, 4, 52]:
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # Rolling features
    for window in [4, 12, 52]:
        df[f'roll_mean_{window}'] = df['search_volume'].shift(1).rolling(window, min_periods=1).mean()
        df[f'roll_std_{window}'] = df['search_volume'].shift(1).rolling(window, min_periods=1).std()

    return df

def sarima_lgbm_hybrid(train_path, output_path, forecast_weeks=None):
    """
    SARIMA + LightGBM hybrid forecasting

    Args:
        train_path: Path to training data
        output_path: Path to save submission
        forecast_weeks: Weeks to forecast
    """
    print("Loading data...")
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Week'])

    if forecast_weeks is None:
        forecast_weeks = pd.to_datetime([
            '2025-07-06', '2025-07-13', '2025-07-20', '2025-07-27',
            '2025-08-03', '2025-08-10', '2025-08-17', '2025-08-24', '2025-08-31'
        ])
    else:
        forecast_weeks = pd.to_datetime(forecast_weeks)

    n_forecast = len(forecast_weeks)
    items = df['item_id'].unique()

    print(f"Processing {len(items)} items...")

    all_predictions = []

    for idx, item_id in enumerate(items):
        if (idx + 1) % 25 == 0:
            print(f"Progress: {idx + 1}/{len(items)}")

        item_data = df[df['item_id'] == item_id].copy().sort_values('date')
        series = item_data['search_volume'].values

        # Fit SARIMA
        sarima_forecast, sarima_fitted = fit_sarima_model(
            series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 52),
            forecast_steps=n_forecast
        )

        # Add SARIMA components to data
        item_data['sarima_fitted'] = sarima_fitted
        item_data['sarima_residual'] = series - sarima_fitted

        # Create features
        item_data = create_features(item_data)

        # Train LightGBM on residuals
        train_clean = item_data.dropna()

        if len(train_clean) < 40:
            # Not enough data, use SARIMA only
            predictions = sarima_forecast
        else:
            feature_cols = [c for c in train_clean.columns
                          if c not in ['id', 'Week', 'item_id', 'Maincategory',
                                      'Subcategory', 'search_volume', 'date']]

            X = train_clean[feature_cols]
            y = train_clean['search_volume']

            # Quick LightGBM model
            model = lgb.LGBMRegressor(
                objective='mape',
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )

            model.fit(X, y)

            # Forecast iteratively
            predictions = []
            curr_data = item_data.copy()

            for i, fdate in enumerate(forecast_weeks):
                # Create forecast row
                frow = pd.DataFrame({
                    'date': [fdate],
                    'Maincategory': [item_data['Maincategory'].iloc[0]],
                    'Subcategory': [item_data['Subcategory'].iloc[0]],
                    'search_volume': [np.nan],
                    'sarima_fitted': [sarima_forecast[i]],
                    'sarima_residual': [0]
                })

                temp_data = pd.concat([curr_data, frow], ignore_index=True)
                temp_data = create_features(temp_data)

                # Get features for prediction
                ffeatures = temp_data.iloc[-1:][feature_cols]

                # Fill NaN
                for col in ffeatures.columns:
                    if ffeatures[col].isna().any():
                        ffeatures[col].fillna(temp_data[col].mean(), inplace=True)

                # Predict
                pred = model.predict(ffeatures)[0]
                pred = max(pred, 0)

                predictions.append(pred)

                # Update for next iteration
                frow['search_volume'] = pred
                curr_data = pd.concat([curr_data, frow], ignore_index=True)

            predictions = np.array(predictions)

        # Store results
        for i, fdate in enumerate(forecast_weeks):
            all_predictions.append({
                'id': f"{fdate.strftime('%Y-%m-%d')}{item_id}",
                'search_volume': max(predictions[i], 0)
            })

    # Save submission
    submission = pd.DataFrame(all_predictions)
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(submission)}")

    return submission

if __name__ == "__main__":
    train_path = "../data/train.csv"
    output_path = "../submissions/sarima_lgbm_submission.csv"

    submission = sarima_lgbm_hybrid(train_path, output_path)
    print("\nSample predictions:")
    print(submission.head(20))

