"""
Training script for Patient Readmission Risk Model with SHAP support.
Uses RandomForestClassifier for tree-based SHAP explanations.

Run this script to train/retrain the model:
    python utils/train_risk_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / 'patient_risk_model.joblib'


def load_data_from_db():
    """Load patient data from database."""
    from file_reader import load_data_from_db as db_loader
    return db_loader()


def train_risk_model():
    """Train the patient readmission risk model with SHAP support."""
    print("Loading data from database...")
    df = load_data_from_db()
    
    if df.empty:
        print("ERROR: Cannot train model - Database is empty.")
        return None
    
    print(f"Loaded {len(df)} records.")
    
    # Define features and target
    features = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
    target = 'Readmission'
    
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train RandomForestClassifier (compatible with SHAP TreeExplainer)
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, model.predict(X_test)))
    
    # Feature importances
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    # Save model with metadata
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': features,
        'trained_on_n': len(df_cleaned),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_version': '2.0-shap',
        'target_distribution': {
            'Yes': int((y == 'Yes').sum()),
            'No': int((y == 'No').sum())
        }
    }
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save the model
    joblib.dump(model_data, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")
    
    return model_data


if __name__ == "__main__":
    print("=" * 60)
    print("Patient Readmission Risk Model Training")
    print("=" * 60)
    train_risk_model()
