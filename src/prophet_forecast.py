import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

FORECAST_WEEKS = [
    '2025-07-06', '2025-07-13', '2025-07-20', '2025-07-27',
    '2025-08-03', '2025-08-10', '2025-08-17', '2025-08-24', '2025-08-31'
]

def load_data(train_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Week'])
    return df

def prepare_prophet_frame(item_df: pd.DataFrame) -> pd.DataFrame:
    # Prophet expects columns ds, y
    return item_df.rename(columns={'date': 'ds', 'search_volume': 'y'})[['ds', 'y']]

def seasonal_naive_fallback(series: np.ndarray, horizon: int, seasonal_periods: int = 52) -> np.ndarray:
    if len(series) >= seasonal_periods:
        last_season = series[-seasonal_periods:]
        reps = int(np.ceil(horizon / seasonal_periods))
        return np.tile(last_season, reps)[:horizon]
    if len(series) == 0:
        return np.zeros(horizon, dtype=float)
    return np.full(horizon, series[-1], dtype=float)

def fit_prophet_and_forecast(df_item: pd.DataFrame, horizon: int) -> np.ndarray:
    try:
        from prophet import Prophet  # local import to allow fallback if not installed
    except Exception:
        return seasonal_naive_fallback(df_item['y'].values.astype(float), horizon)

    if len(df_item) < 5:  # too few points for Prophet
        return seasonal_naive_fallback(df_item['y'].values.astype(float), horizon)

    # Configure Prophet (weekly data -> yearly seasonality helpful, weekly seasonality redundant)
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    # Add explicit yearly seasonality if desired; Prophet defaults to it when yearly_seasonality=True.
    try:
        model.fit(df_item)
        future = model.make_future_dataframe(periods=horizon, freq='W-SUN')
        forecast = model.predict(future)
        fc_tail = forecast.tail(horizon)['yhat'].values.astype(float)
        return fc_tail
    except Exception:
        return seasonal_naive_fallback(df_item['y'].values.astype(float), horizon)

def prophet_forecast(train_path: str, output_path: str):
    df = load_data(train_path)
    forecast_weeks = pd.to_datetime(FORECAST_WEEKS)
    items = df['item_id'].unique()
    submission_rows = []
    horizon = len(forecast_weeks)

    print(f"Training Prophet (or fallback) on {len(items)} items...")
    for idx, item in enumerate(items):
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(items)} items...")
        item_df = df[df['item_id'] == item].sort_values('date')
        prep = prepare_prophet_frame(item_df)
        fc_values = fit_prophet_and_forecast(prep, horizon)
        # clamp and round
        fc_values = np.clip(np.round(fc_values), 0, 100)
        for w, val in zip(forecast_weeks, fc_values):
            submission_rows.append({
                'id': f"{w.strftime('%Y-%m-%d')}{int(item)}",
                'Week': w.strftime('%d-%m-%Y'),
                'item_id': int(item),
                'search_volume_forecast': int(val)
            })

    submission_df = pd.DataFrame(submission_rows).sort_values('id').reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)
    print(f"\nProphet Forecast Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)} | Items: {submission_df['item_id'].nunique()} | Weeks: {horizon}")
    print(submission_df.head(10))

if __name__ == '__main__':
    train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    out_file = os.path.join(os.path.dirname(__file__), '..', 'submissions', 'prophet_submission.csv')
    prophet_forecast(train_file, out_file)
