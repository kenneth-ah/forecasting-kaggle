import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(train_path):
    """Load raw training data"""
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Week'])
    return df

def handle_null_values(df):
    """Handle null/missing values in the dataframe"""
    print("Handling null values...")
    
    initial_nulls = df['search_volume'].isnull().sum()
    if initial_nulls > 0:
        print(f"  Found {initial_nulls} null values in search_volume")
        
        # Group by item_id and fill nulls with forward fill then backward fill
        df['search_volume'] = df.groupby('item_id')['search_volume'].fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining nulls, use item mean
        remaining_nulls = df['search_volume'].isnull().sum()
        if remaining_nulls > 0:
            item_means = df.groupby('item_id')['search_volume'].mean()
            df['search_volume'] = df.apply(
                lambda row: item_means[row['item_id']] if pd.isnull(row['search_volume']) else row['search_volume'],
                axis=1
            )
        
        # For any still-remaining nulls, use global mean
        if df['search_volume'].isnull().sum() > 0:
            global_mean = df['search_volume'].mean()
            df['search_volume'] = df['search_volume'].fillna(global_mean)
        
        print(f"  Filled all null values (used forward fill, then item mean, then global mean)")
    
    return df

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def preprocess_data(train_path, output_dir, seq_length=4):
    """
    Preprocess training data and save normalized sequences.
    
    Args:
        train_path: Path to raw training data CSV
        output_dir: Directory to save preprocessed data
        seq_length: Length of sequences for LSTM (default: 4)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading raw data...")
    df = load_raw_data(train_path)
    
    # Handle null values
    df = handle_null_values(df)
    
    # Sort by date
    df = df.sort_values('date')
    
    items = df['item_id'].unique()
    preprocessed_data = {}
    scalers = {}
    statistics = []
    
    print(f"Preprocessing {len(items)} items...")
    
    for idx, item_id in enumerate(items):
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(items)} items...")
        
        item_data = df[df['item_id'] == item_id].copy()
        item_data = item_data.sort_values('date')
        
        # Extract historical values
        historical_values = item_data['search_volume'].values.astype(np.float32)
        
        # Calculate statistics
        stats = {
            'item_id': int(item_id),
            'data_points': len(historical_values),
            'mean': float(historical_values.mean()),
            'std': float(historical_values.std()),
            'min': float(historical_values.min()),
            'max': float(historical_values.max()),
        }
        statistics.append(stats)
        
        if len(historical_values) < seq_length:
            print(f"  Warning: Item {item_id} has insufficient data ({len(historical_values)} < {seq_length})")
            continue
        
        try:
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(historical_values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_train, y_train = create_sequences(scaled_data, seq_length)
            
            if len(X_train) < 2:
                print(f"  Warning: Item {item_id} has insufficient sequences ({len(X_train)} < 2)")
                continue
            
            # Reshape for LSTM [samples, timesteps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Store preprocessed data
            preprocessed_data[int(item_id)] = {
                'X_train': X_train,
                'y_train': y_train,
                'historical_values': historical_values,
                'last_sequence': scaled_data[-seq_length:],
                'dates': item_data['date'].values,
            }
            
            # Store scaler
            scalers[int(item_id)] = scaler
            
        except Exception as e:
            print(f"  Error preprocessing item {item_id}: {str(e)}")
            continue
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    
    # Save sequences and scalers
    with open(os.path.join(output_dir, 'preprocessed_sequences.pkl'), 'wb') as f:
        pickle.dump(preprocessed_data, f)
    print(f"  Saved preprocessed sequences: {len(preprocessed_data)} items")
    
    # Save scalers
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    print(f"  Saved scalers: {len(scalers)} items")
    
    # Save statistics
    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv(os.path.join(output_dir, 'data_statistics.csv'), index=False)
    print(f"  Saved statistics: {len(statistics)} items")
    
    # Save configuration
    config = {
        'seq_length': seq_length,
        'preprocessing_date': datetime.now().isoformat(),
        'total_items': len(items),
        'preprocessed_items': len(preprocessed_data),
        'total_scalers': len(scalers),
    }
    
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved configuration")
    
    print(f"\nPreprocessing complete!")
    print(f"  Total items: {len(items)}")
    print(f"  Successfully preprocessed: {len(preprocessed_data)}")
    print(f"  Failed items: {len(items) - len(preprocessed_data)}")
    print(f"  Sequence length: {seq_length}")
    
    return preprocessed_data, scalers, stats_df

def load_preprocessed_data(output_dir):
    """Load previously preprocessed data"""
    print(f"Loading preprocessed data from {output_dir}...")
    
    with open(os.path.join(output_dir, 'preprocessed_sequences.pkl'), 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    with open(os.path.join(output_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    
    stats_df = pd.read_csv(os.path.join(output_dir, 'data_statistics.csv'))
    
    print(f"Loaded {len(preprocessed_data)} preprocessed items")
    print(f"Loaded {len(scalers)} scalers")
    
    return preprocessed_data, scalers, stats_df

if __name__ == "__main__":
    train_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "train.csv"
    )
    
    output_directory = os.path.join(
        os.path.dirname(__file__),
        "..",
        "preprocessed_data"
    )
    
    # Preprocess the data
    preprocessed_data, scalers, stats_df = preprocess_data(train_file, output_directory, seq_length=4)
    
    print("\nData Statistics Summary:")
    print(stats_df.describe())
