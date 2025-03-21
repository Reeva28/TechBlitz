import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load and preprocess the air quality data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed data
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Check for missing values and fill if necessary
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())
    
    return df

def get_basic_stats(df):
    """
    Get basic statistics for the dataset.
    
    Args:
        df: pandas.DataFrame containing the data
        
    Returns:
        pandas.DataFrame: Statistics for each column
    """
    # Calculate statistics
    stats = pd.DataFrame({
        'Mean': df.mean(),
        'Median': df.median(),
        'Std Dev': df.std(),
        'Min': df.min(),
        'Max': df.max()
    })
    
    return stats

def prepare_data(df, target_variable, exclude_features=None, test_size=0.2, random_state=42):
    """
    Prepare data for model training.
    
    Args:
        df: pandas.DataFrame containing the data
        target_variable: Target variable for prediction
        exclude_features: List of features to exclude
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X, y, X_train, X_test, y_train, y_test)
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Define target variables
    target_vars = ["Air Quality_Hazardous", "Air Quality_Moderate", "Air Quality_Poor"]
    
    # Set features and target
    X = data.drop(columns=target_vars)
    y = data[target_variable]
    
    # Exclude features if specified
    if exclude_features and len(exclude_features) > 0:
        X = X.drop(columns=exclude_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test
