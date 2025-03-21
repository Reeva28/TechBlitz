import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgboost_model(X_train, y_train, params=None):
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: XGBoost parameters
        
    Returns:
        tuple: (trained_model, training_time)
    """
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # Update parameters if provided
    if params:
        default_params.update(params)
    
    # Create model
    model = xgb.XGBClassifier(**default_params)
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

def predict_with_model(model, X_test):
    """
    Make predictions with the trained model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        
    Returns:
        numpy.ndarray: Predictions
    """
    return model.predict(X_test)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        pandas.DataFrame: Feature importance
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance
