import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_percentage_error

def create_comprehensive_features(df, item_id, main_category, sub_category):
    """Create comprehensive feature set for time series forecasting"""
    df = df.copy()

    # ========== Time-based features ==========
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Summer months indicator (July-August)
    df['is_summer'] = df['month'].isin([7, 8]).astype(int)
    df['is_july'] = (df['month'] == 7).astype(int)
    df['is_august'] = (df['month'] == 8).astype(int)

    # ========== Lag features ==========
    # Short-term lags (recent weeks)
    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # Medium-term lags (monthly)
    for lag in [8, 12, 16]:
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # Seasonal lags (yearly)
    for lag in [52, 51, 53]:  # Same week last year ± 1 week
        df[f'lag_{lag}'] = df['search_volume'].shift(lag)

    # ========== Rolling statistics ==========
    # Short-term windows
    for window in [2, 4, 8]:
        df[f'rolling_mean_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).min()

    # Medium-term windows
    for window in [12, 26]:
        df[f'rolling_mean_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'rolling_median_{window}'] = df['search_volume'].shift(1).rolling(window=window, min_periods=1).median()

    # Long-term window (yearly)
    df['rolling_mean_52'] = df['search_volume'].shift(1).rolling(window=52, min_periods=1).mean()
    df['rolling_std_52'] = df['search_volume'].shift(1).rolling(window=52, min_periods=1).std()

    # ========== Exponential weighted features ==========
    for span in [4, 8, 12, 26]:
        df[f'ewm_{span}'] = df['search_volume'].shift(1).ewm(span=span, adjust=False).mean()

    # ========== Momentum and trend features ==========
    df['momentum_2'] = df['search_volume'].shift(1) - df['search_volume'].shift(3)
    df['momentum_4'] = df['search_volume'].shift(1) - df['search_volume'].shift(5)
    df['momentum_8'] = df['search_volume'].shift(1) - df['search_volume'].shift(9)
    df['momentum_12'] = df['search_volume'].shift(1) - df['search_volume'].shift(13)

    # Year-over-year changes
    df['yoy_change'] = df['search_volume'].shift(1) - df['search_volume'].shift(53)
    df['yoy_pct_change'] = df['yoy_change'] / (df['search_volume'].shift(53) + 1)

    # ========== Volatility features ==========
    df['volatility_4'] = df['search_volume'].shift(1).rolling(window=4, min_periods=1).std() / (df['search_volume'].shift(1).rolling(window=4, min_periods=1).mean() + 1)
    df['volatility_12'] = df['search_volume'].shift(1).rolling(window=12, min_periods=1).std() / (df['search_volume'].shift(1).rolling(window=12, min_periods=1).mean() + 1)

    # ========== Expanding features ==========
    df['expanding_mean'] = df['search_volume'].shift(1).expanding(min_periods=1).mean()
    df['expanding_std'] = df['search_volume'].shift(1).expanding(min_periods=1).std()

    # ========== Ratios and differences ==========
    df['ratio_to_rolling_mean_4'] = df['search_volume'].shift(1) / (df['rolling_mean_4'] + 1)
    df['ratio_to_rolling_mean_12'] = df['search_volume'].shift(1) / (df['rolling_mean_12'] + 1)
    df['ratio_to_rolling_mean_52'] = df['search_volume'].shift(1) / (df['rolling_mean_52'] + 1)

    # ========== Seasonal decomposition features ==========
    # Simple seasonal average (same week of year)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    seasonal_avg = df.groupby('week_of_year')['search_volume'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df['seasonal_avg'] = seasonal_avg
    df['ratio_to_seasonal'] = df['search_volume'].shift(1) / (seasonal_avg + 1)

    # ========== Static features ==========
    df['item_id'] = item_id
    df['main_category'] = main_category
    df['sub_category'] = sub_category

    return df

def create_category_features(df):
    """Create category-level aggregate features"""
    # Subcategory aggregates
    cat_agg = df.groupby(['date', 'Subcategory'])['search_volume'].agg([
        ('cat_mean', 'mean'),
        ('cat_median', 'median'),
        ('cat_std', 'std'),
        ('cat_max', 'max'),
        ('cat_min', 'min')
    ]).reset_index()

    df = df.merge(cat_agg, on=['date', 'Subcategory'], how='left')

    # Ratio to category
    df['ratio_to_cat_mean'] = df['search_volume'] / (df['cat_mean'] + 1)
    df['diff_from_cat_median'] = df['search_volume'] - df['cat_median']

    # Main category aggregates
    main_cat_agg = df.groupby(['date', 'Maincategory'])['search_volume'].agg([
        ('main_cat_mean', 'mean'),
        ('main_cat_median', 'median')
    ]).reset_index()

    df = df.merge(main_cat_agg, on=['date', 'Maincategory'], how='left')
    df['ratio_to_main_cat_mean'] = df['search_volume'] / (df['main_cat_mean'] + 1)

    return df

def catboost_advanced_forecast(train_path, output_path, forecast_weeks=None):
    """
    CatBoost with advanced feature engineering for time series forecasting.

    Key improvements over Holt-Winters + LightGBM:
    1. CatBoost handles categorical features natively and often performs better
    2. More comprehensive feature engineering
    3. Separate models for different item clusters based on variance
    4. Better handling of seasonal patterns

    Args:
        train_path: Path to training data CSV
        output_path: Path where the submission CSV will be saved
        forecast_weeks: List of weeks to forecast
    """

    print("Loading training data...")
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Week'])

    # Define forecast weeks
    if forecast_weeks is None:
        forecast_weeks = [
            '2025-07-06', '2025-07-13', '2025-07-20', '2025-07-27',
            '2025-08-03', '2025-08-10', '2025-08-17', '2025-08-24', '2025-08-31'
        ]

    forecast_weeks = pd.to_datetime(forecast_weeks)
    n_forecast_weeks = len(forecast_weeks)

    print(f"Forecasting for {n_forecast_weeks} weeks from {forecast_weeks[0]} to {forecast_weeks[-1]}")

    # Create category-level features
    print("Creating category-level features...")
    df = create_category_features(df)

    items = df['item_id'].unique()
    print(f"Number of items: {len(items)}")

    all_predictions = []

    # Process each item
    for idx, item_id in enumerate(items):
        if (idx + 1) % 20 == 0:
            print(f"Processing item {idx + 1}/{len(items)}")

        # Filter data for this item
        item_data = df[df['item_id'] == item_id].copy()
        item_data = item_data.sort_values('date').reset_index(drop=True)

        main_category = item_data['Maincategory'].iloc[0]
        sub_category = item_data['Subcategory'].iloc[0]

        # Create features
        item_data = create_comprehensive_features(item_data, item_id, main_category, sub_category)

        # Remove rows with NaN values
        train_data = item_data.dropna()

        if len(train_data) < 30:
            # Not enough data, use simple seasonal naive
            predictions = []
            for forecast_date in forecast_weeks:
                same_week_last_year = forecast_date - pd.DateOffset(years=1)
                historical = item_data[item_data['date'] == same_week_last_year]
                if len(historical) > 0:
                    pred = historical['search_volume'].values[0]
                else:
                    pred = item_data['search_volume'].mean()
                predictions.append(pred)
            predictions = np.array(predictions)
        else:
            # Define features and target
            exclude_cols = ['id', 'Week', 'item_id', 'Maincategory', 'Subcategory',
                          'search_volume', 'date', 'week_of_year']
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]

            # Identify categorical features
            cat_features = ['main_category', 'sub_category', 'month', 'quarter',
                          'is_summer', 'is_july', 'is_august']
            cat_features = [f for f in cat_features if f in feature_cols]

            X_train = train_data[feature_cols]
            y_train = train_data['search_volume']

            # Check if target has variation
            if y_train.std() == 0 or y_train.nunique() == 1:
                # Constant target, use mean as prediction
                predictions = np.array([y_train.mean()] * n_forecast_weeks)
                # Store and continue
                for i, forecast_date in enumerate(forecast_weeks):
                    week_str = forecast_date.strftime('%Y-%m-%d')
                    submission_id = f"{week_str}{item_id}"
                    all_predictions.append({
                        'id': submission_id,
                        'search_volume': predictions[i]
                    })
                continue

            # Calculate item-specific coefficient of variation
            cv = y_train.std() / (y_train.mean() + 1)

            # Adjust model parameters based on item characteristics
            if cv > 0.5:  # High variance item
                iterations = 300
                learning_rate = 0.03
                depth = 7
            elif cv > 0.3:  # Medium variance
                iterations = 250
                learning_rate = 0.05
                depth = 6
            else:  # Low variance
                iterations = 200
                learning_rate = 0.05
                depth = 5

            # Train CatBoost model
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function='MAPE',  # Directly optimize MAPE
                eval_metric='MAPE',
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                cat_features=cat_features,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                rsm=0.8,  # Random subspace method
                border_count=128
            )

            # Use last 15% as validation
            split_point = int(len(X_train) * 0.85)
            X_train_split = X_train.iloc[:split_point]
            y_train_split = y_train.iloc[:split_point]
            X_val = X_train.iloc[split_point:]
            y_val = y_train.iloc[split_point:]

            # Check if both splits have variance
            train_has_variance = (len(y_train_split) > 5 and y_train_split.std() > 0 and y_train_split.nunique() > 1)
            val_has_variance = (len(y_val) > 5 and y_val.std() > 0 and y_val.nunique() > 1)
            use_validation = train_has_variance and val_has_variance

            if use_validation:
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=30,
                    verbose=False
                )
            else:
                # Train on full data without validation
                model.fit(X_train, y_train, verbose=False)

            # Generate forecasts iteratively
            predictions = []
            current_data = item_data.copy()

            for i, forecast_date in enumerate(forecast_weeks):
                # Create forecast row
                forecast_row = pd.DataFrame({
                    'date': [forecast_date],
                    'search_volume': [np.nan],  # Will be filled
                    'Maincategory': [main_category],
                    'Subcategory': [sub_category]
                })

                # Add to current data
                temp_data = pd.concat([current_data, forecast_row], ignore_index=True)

                # Recreate features (this will use predictions as historical data)
                temp_data = create_comprehensive_features(temp_data, item_id, main_category, sub_category)

                # Get the last row features
                forecast_features = temp_data.iloc[-1:][feature_cols]

                # Fill any remaining NaN with reasonable defaults
                for col in forecast_features.columns:
                    if forecast_features[col].isna().any():
                        if col in cat_features:
                            forecast_features[col] = forecast_features[col].ffill()
                        else:
                            # Use mean of recent values
                            recent_mean = temp_data[col].tail(10).mean()
                            if pd.isna(recent_mean):
                                recent_mean = 0
                            forecast_features[col] = forecast_features[col].fillna(recent_mean)

                # Make prediction
                pred = model.predict(forecast_features)[0]
                pred = max(pred, 0)  # Ensure non-negative

                predictions.append(pred)

                # Update current_data with prediction
                forecast_row['search_volume'] = pred

                # Also update category aggregates (simple approach: use last known values)
                for col in ['cat_mean', 'cat_median', 'cat_std', 'cat_max', 'cat_min',
                           'main_cat_mean', 'main_cat_median']:
                    if col in current_data.columns:
                        forecast_row[col] = current_data[col].iloc[-1]

                current_data = pd.concat([current_data, forecast_row], ignore_index=True)

            predictions = np.array(predictions)

        # Store predictions
        for i, forecast_date in enumerate(forecast_weeks):
            week_str = forecast_date.strftime('%Y-%m-%d')
            submission_id = f"{week_str}{item_id}"

            all_predictions.append({
                'id': submission_id,
                'search_volume': predictions[i]
            })

    # Create submission
    print("\nCreating submission file...")
    submission_df = pd.DataFrame(all_predictions)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")

    return submission_df

if __name__ == "__main__":
    train_path = "../data/train.csv"
    output_path = "../submissions/catboost_advanced_submission.csv"

    submission = catboost_advanced_forecast(train_path, output_path)

    print("\nSample predictions:")
    print(submission.head(20))

