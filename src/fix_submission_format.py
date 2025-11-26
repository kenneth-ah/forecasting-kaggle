import pandas as pd

# Files to fix
files_to_fix = [
    'submissions/ensemble_submission.csv',
    'submissions/ensemble_equal_submission.csv',
    'submissions/prophet_xgboost_submission.csv',
    'submissions/catboost_advanced_submission.csv',
    'submissions/sarima_lgbm_submission.csv'
]

for filepath in files_to_fix:
    try:
        # Read the file
        df = pd.read_csv(filepath)

        print(f'\nProcessing {filepath}...')
        print(f'  Current columns: {df.columns.tolist()}')
        print(f'  Sample ID before: {df["id"].iloc[0]}')

        # The id format is like "2025-07-061" where:
        # - "2025-07-06" is the date
        # - "1" at the end is the item_id

        # Extract the date part (first 10 characters: YYYY-MM-DD)
        df['date_part'] = df['id'].str[:10]

        # Extract item_id (everything after the date)
        df['item_id'] = df['id'].str[10:].astype(int)

        # Convert date to Week format (d-m-yyyy)
        df['Week'] = pd.to_datetime(df['date_part']).dt.strftime('%-d-%-m-%Y')

        # Rename search_volume to search_volume_forecast if needed
        if 'search_volume' in df.columns:
            df = df.rename(columns={'search_volume': 'search_volume_forecast'})

        # Reorder columns to match expected format
        df = df[['id', 'Week', 'item_id', 'search_volume_forecast']]

        # Save back
        df.to_csv(filepath, index=False)

        print(f'  ✅ Fixed! New columns: {df.columns.tolist()}')
        print(f'  Sample after:')
        print(f'    ID: {df["id"].iloc[0]}')
        print(f'    Week: {df["Week"].iloc[0]}')
        print(f'    item_id: {df["item_id"].iloc[0]}')
        print(f'    search_volume_forecast: {df["search_volume_forecast"].iloc[0]:.2f}')

    except Exception as e:
        print(f'  ❌ Error: {e}')
        import traceback
        traceback.print_exc()

print('\n' + '='*60)
print('✅ All files fixed!')
print('='*60)

# Verify the first file matches sample format
print('\nVerifying against sample submission...')
sample = pd.read_csv('submissions/sample_submission.csv')
ensemble = pd.read_csv('submissions/ensemble_submission.csv')

print(f'\nSample columns: {sample.columns.tolist()}')
print(f'Ensemble columns: {ensemble.columns.tolist()}')

if list(sample.columns) == list(ensemble.columns):
    print('✅ Column names match!')
else:
    print('⚠️ Column names do not match!')

print(f'\nSample first row:')
print(sample.head(1))
print(f'\nEnsemble first row:')
print(ensemble.head(1))

