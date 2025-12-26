"""
Training script for Length of Stay (LOS) Prediction Model with SHAP support.
Uses RandomForestRegressor for tree-based SHAP explanations.

Run this script to train/retrain the model:
    python utils/train_los_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / 'patient_los_model.joblib'


def load_data_from_db():
    """Load patient data from database."""
    from file_reader import load_data_from_db as db_loader
    return db_loader()


def train_los_model():
    """Train the Length of Stay prediction model with SHAP support."""
    print("Loading data from database...")
    df = load_data_from_db()
    
    if df.empty:
        print("ERROR: Cannot train model - Database is empty.")
        return None
    
    print(f"Loaded {len(df)} records.")
    
    # Define features and target
    # Including Cost as a feature for LOS prediction
    features = ['Age', 'Gender', 'Condition', 'Procedure', 'Cost']
    target = 'Length_of_Stay'
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Drop rows with NaN in features or target
    df_cleaned = df.dropna(subset=features + [target])
    if df_cleaned.empty:
        print("ERROR: No complete data rows after dropping NaNs.")
        return None
    
    print(f"Using {len(df_cleaned)} records after cleaning.")
    
    # Prepare data
    X = df_cleaned[features].copy()
    y = df_cleaned[target]
    
    # Encode categorical variables using LabelEncoder
    # This is required for SHAP TreeExplainer to work properly
    label_encoders = {}
    categorical_features = ['Gender', 'Condition', 'Procedure']
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train RandomForestRegressor (compatible with SHAP TreeExplainer)
    print("\nTraining RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\nModel Performance:")
    print(f"  Train MAE: {train_mae:.2f} days")
    print(f"  Test MAE: {test_mae:.2f} days")
    print(f"  Train RMSE: {train_rmse:.2f} days")
    print(f"  Test RMSE: {test_rmse:.2f} days")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Feature importances
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    # Target statistics for context
    target_stats = {
        'mean': float(y.mean()),
        'std': float(y.std()),
        'min': float(y.min()),
        'max': float(y.max()),
        'median': float(y.median())
    }
    
    print(f"\nTarget Statistics (Length_of_Stay):")
    print(f"  Mean: {target_stats['mean']:.2f} days")
    print(f"  Std: {target_stats['std']:.2f} days")
    print(f"  Range: {target_stats['min']:.0f} - {target_stats['max']:.0f} days")
    
    # Save model with metadata
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': features,
        'trained_on_n': len(df_cleaned),
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'model_version': '1.0-shap',
        'target_stats': target_stats
    }
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save the model
    joblib.dump(model_data, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")
    
    return model_data


if __name__ == "__main__":
    print("=" * 60)
    print("Length of Stay (LOS) Prediction Model Training")
    print("=" * 60)
    train_los_model()

