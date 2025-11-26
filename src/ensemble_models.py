"""
Ensemble multiple forecasting models to improve predictions.

This script combines predictions from different models using weighted averaging.
Ensembling often reduces variance and improves overall performance.
"""

import pandas as pd
import numpy as np
import os

def ensemble_predictions(model_files, weights=None, output_path=None):
    """
    Ensemble multiple model predictions using weighted averaging.

    Args:
        model_files: Dictionary mapping model names to file paths
        weights: Dictionary mapping model names to weights (if None, equal weights)
        output_path: Path to save ensemble submission

    Returns:
        DataFrame with ensemble predictions
    """

    # Load all predictions
    predictions = {}
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Standardize column name
            if 'search_volume_forecast' in df.columns:
                df = df.rename(columns={'search_volume_forecast': 'search_volume'})
            if 'search_volume' not in df.columns:
                print(f"Warning: {filepath} missing search_volume column, skipping {name}")
                continue
            predictions[name] = df[['id', 'search_volume']]
            print(f"Loaded {name}: {len(predictions[name])} predictions")
        else:
            print(f"Warning: {filepath} not found, skipping {name}")

    if not predictions:
        raise ValueError("No valid prediction files found!")

    # Set equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
    else:
        # Normalize weights
        total = sum(weights.values())
        weights = {name: w / total for name, w in weights.items()}

    print("\nEnsemble weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")

    # Create ensemble by weighted averaging
    # Start with the first model as base
    base_name = list(predictions.keys())[0]
    ensemble_df = predictions[base_name][['id']].copy()

    # Calculate weighted average of search_volume
    ensemble_df['search_volume'] = 0.0

    for name, df in predictions.items():
        if name in weights:
            # Merge on id to ensure alignment
            merged = ensemble_df[['id']].merge(df[['id', 'search_volume']], on='id', how='left')
            ensemble_df['search_volume'] += weights[name] * merged['search_volume'].fillna(0)

    # Ensure non-negative predictions
    ensemble_df['search_volume'] = ensemble_df['search_volume'].clip(lower=0)

    # Save if output path provided
    if output_path:
        ensemble_df.to_csv(output_path, index=False)
        print(f"\nEnsemble saved to {output_path}")
        print(f"Total predictions: {len(ensemble_df)}")

    return ensemble_df

def compare_predictions(model_files):
    """
    Compare predictions across models to understand differences.

    Args:
        model_files: Dictionary mapping model names to file paths
    """

    # Load all predictions
    predictions = {}
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Standardize column name
            if 'search_volume_forecast' in df.columns:
                df = df.rename(columns={'search_volume_forecast': 'search_volume'})
            if 'search_volume' not in df.columns:
                print(f"Warning: {filepath} missing search_volume column, skipping")
                continue
            predictions[name] = df[['id', 'search_volume']]

    if len(predictions) < 2:
        print("Need at least 2 models to compare")
        return

    # Extract item IDs from the first prediction
    base_name = list(predictions.keys())[0]
    base_df = predictions[base_name]

    # Add a column for item_id (extract from id)
    base_df['item_id'] = base_df['id'].str.extract(r'(\d+)$')[0].astype(int)

    print("\nPrediction Statistics by Model:")
    print("=" * 70)

    for name, df in predictions.items():
        print(f"\n{name}:")
        print(f"  Mean: {df['search_volume'].mean():.2f}")
        print(f"  Median: {df['search_volume'].median():.2f}")
        print(f"  Std: {df['search_volume'].std():.2f}")
        print(f"  Min: {df['search_volume'].min():.2f}")
        print(f"  Max: {df['search_volume'].max():.2f}")

    # Compare correlations
    if len(predictions) >= 2:
        print("\n\nPairwise Correlations:")
        print("=" * 70)

        model_names = list(predictions.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                df1 = predictions[name1].sort_values('id')
                df2 = predictions[name2].sort_values('id')

                corr = df1['search_volume'].corr(df2['search_volume'])
                print(f"  {name1} vs {name2}: {corr:.4f}")

if __name__ == "__main__":

    # Define model files
    model_files = {
        'Holt-Winters + LightGBM': '../submissions/holtwinters_lightgbm_submission.csv',
        'Prophet + XGBoost': '../submissions/prophet_xgboost_submission.csv',
        'CatBoost Advanced': '../submissions/catboost_advanced_submission.csv',
        'SARIMA + LightGBM': '../submissions/sarima_lgbm_submission.csv'
    }

    # Compare predictions
    print("Comparing model predictions...")
    compare_predictions(model_files)

    # Create ensemble with optimized weights
    # Give more weight to models that typically perform better on Google Trends data
    weights = {
        'Prophet + XGBoost': 0.35,  # Strong on seasonality
        'CatBoost Advanced': 0.35,  # Direct MAPE optimization
        'SARIMA + LightGBM': 0.20,  # Good statistical baseline
        'Holt-Winters + LightGBM': 0.10  # Current baseline
    }

    print("\n\n" + "=" * 70)
    print("Creating Ensemble Model...")
    print("=" * 70)

    ensemble_df = ensemble_predictions(
        model_files,
        weights=weights,
        output_path='../submissions/ensemble_submission.csv'
    )

    print("\n\nSample ensemble predictions:")
    print(ensemble_df.head(20))

    # Also create equal-weight ensemble
    print("\n\n" + "=" * 70)
    print("Creating Equal-Weight Ensemble...")
    print("=" * 70)

    ensemble_equal = ensemble_predictions(
        model_files,
        weights=None,
        output_path='../submissions/ensemble_equal_submission.csv'
    )

    print("\n✅ Ensemble models created successfully!")
    print("\nGenerated submissions:")
    print("  1. ensemble_submission.csv (optimized weights)")
    print("  2. ensemble_equal_submission.csv (equal weights)")

