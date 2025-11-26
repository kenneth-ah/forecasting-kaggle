import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def lstm_forecast(preprocessed_data_path, scalers_path, output_path, forecast_weeks=None):
    """
    LSTM-based forecasting model using preprocessed data.
    
    Args:
        preprocessed_data_path: Path to preprocessed_sequences.pkl
        scalers_path: Path to scalers.pkl
        output_path: Path where the submission CSV will be saved
        forecast_weeks: List of weeks to forecast
    """
    
    # Load preprocessed data and scalers
    print("Loading preprocessed data...")
    with open(preprocessed_data_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Define forecast weeks for July-August 2025
    if forecast_weeks is None:
        forecast_weeks = [
            '2025-07-06', '2025-07-13', '2025-07-20', '2025-07-27',
            '2025-08-03', '2025-08-10', '2025-08-17', '2025-08-24', '2025-08-31'
        ]
    
    forecast_weeks = pd.to_datetime(forecast_weeks)
    
    submission_rows = []
    items = list(preprocessed_data.keys())
    
    print(f"Training LSTM model on {len(items)} items...")
    
    for idx, item_id in enumerate(items):
        if (idx + 1) % 20 == 0:
            print(f"  Processing item {idx + 1}/{len(items)}...")
        
        item_id_int = int(item_id)
        
        # Skip if item not in preprocessed data
        if item_id not in preprocessed_data:
            continue
        
        try:
            # Get preprocessed data
            X_train = preprocessed_data[item_id]['X_train']
            y_train = preprocessed_data[item_id]['y_train']
            last_sequence = preprocessed_data[item_id]['last_sequence']
            historical_values = preprocessed_data[item_id]['historical_values']
            scaler = scalers[item_id]
            
            if len(X_train) < 2:
                # Not enough sequences, use average
                item_avg = historical_values.mean()
                for forecast_date in forecast_weeks:
                    submission_rows.append({
                        'id': f"{forecast_date.strftime('%Y-%m-%d')}{item_id_int}",
                        'Week': forecast_date.strftime('%d-%m-%Y'),
                        'item_id': item_id_int,
                        'search_volume_forecast': max(0, min(100, int(item_avg)))
                    })
                continue
            
            # Build LSTM model
            model = Sequential([
                LSTM(32, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False),
                Dropout(0.1),
                Dense(8, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            
            # Train model (quick training)
            model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)
            
            # Generate forecasts for each week
            current_sequence = last_sequence.copy()
            
            for forecast_date in forecast_weeks:
                # Predict next value
                X_pred = current_sequence.reshape(1, len(current_sequence), 1)
                scaled_pred = model.predict(X_pred, verbose=0)[0, 0]
                
                # Inverse transform
                forecast_value = scaler.inverse_transform([[scaled_pred]])[0, 0]
                forecast_value = max(0, min(100, int(forecast_value)))
                
                submission_rows.append({
                    'id': f"{forecast_date.strftime('%Y-%m-%d')}{item_id_int}",
                    'Week': forecast_date.strftime('%d-%m-%Y'),
                    'item_id': item_id_int,
                    'search_volume_forecast': forecast_value
                })
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], scaled_pred)
            
            # Clean up model
            keras.backend.clear_session()
            
        except Exception as e:
            print(f"  Warning: Error processing item {item_id}: {str(e)}")
            # Fallback to average
            if item_id in preprocessed_data:
                item_avg = preprocessed_data[item_id]['historical_values'].mean()
                for forecast_date in forecast_weeks:
                    submission_rows.append({
                        'id': f"{forecast_date.strftime('%Y-%m-%d')}{item_id_int}",
                        'Week': forecast_date.strftime('%d-%m-%Y'),
                        'item_id': item_id_int,
                        'search_volume_forecast': max(0, min(100, int(item_avg)))
                    })
    
    # Create and save submission
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nLSTM Forecast Submission saved to {output_path}")
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
    preprocessed_data_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "preprocessed_data",
        "preprocessed_sequences.pkl"
    )
    
    scalers_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "preprocessed_data",
        "scalers.pkl"
    )
    
    output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "submissions",
        "lstm_submission.csv"
    )
    
    lstm_forecast(preprocessed_data_file, scalers_file, output_file)
