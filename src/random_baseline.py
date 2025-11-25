import pandas as pd
import numpy as np
import os

def generate_random_submission(output_path):
    """
    Generate a simple random baseline submission for the forecasting competition.
    Generates random search volume forecasts for all items and weeks.
    
    Args:
        output_path: Path where the submission CSV will be saved
    """
    # Read the sample submission to get the structure
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "submissions",
        "sample_submission.csv"
    )
    
    df = pd.read_csv(sample_path)
    
    # Generate random search volumes between 0 and 100
    # This is a simple baseline - just random values
    df['search_volume_forecast'] = np.random.randint(0, 100, size=len(df))
    
    # Save the submission
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")

if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "submissions",
        "random_submission.csv"
    )
    generate_random_submission(output_file)
