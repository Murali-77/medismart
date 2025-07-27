import pandas as pd
import joblib # For saving/loading the trained model
import os
from pathlib import Path

# Correct path to model file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / 'patient_risk_model.joblib'

def _load_model():
    """Loads the pre-trained model, or trains it if not found."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)
    else:
        print("Model not found. Training and saving a new model...")
        return _train_and_save_model()

def _train_and_save_model():
    """Train a simple model using the existing hospital data."""
    from .file_reader import load_data_from_csv
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = load_data_from_csv()
    if df.empty:
        print("No data available for training.")
        return None
    
    # Prepare features and target
    features = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
    target = 'Readmission'
    
    # Check if all required columns exist
    if not all(col in df.columns for col in features + [target]):
        print("Required columns missing for training.")
        return None
    
    # Prepare data
    X = df[features].copy()
    y = df[target]
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['Gender', 'Condition', 'Procedure']:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': features
    }
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(model_data, MODEL_FILE)
    print(f"Model trained and saved to {MODEL_FILE}")
    
    return model_data

# Load the model once when the script starts
global_patient_risk_model = _load_model()

def predict_patient_outcome_tool_csv(patient_attributes: dict) -> dict:
    """
    MCP Tool: Patient Risk Assessment (CSV-based)
    Predicts readmission risk for a patient based on provided attributes.

    Args:
        patient_attributes (dict): A dictionary with patient's 'Age', 'Gender',
                                   'Condition', 'Procedure', 'Length_of_Stay'.

    Returns:
        dict: A dictionary with prediction ('Yes'/'No') and probability,
              or an error message.
    """
    if global_patient_risk_model is None:
        return {"status": "error", "message": "Prediction model not available. Please train it first."}

    # Ensure all required attributes are present
    required_attrs = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
    if not all(attr in patient_attributes for attr in required_attrs):
        return {"status": "error", "message": f"Missing required patient attributes. Needed: {required_attrs}"}

    try:
        # Prepare input data
        input_data = patient_attributes.copy()
        
        # If model_data is a dict (new format)
        if isinstance(global_patient_risk_model, dict):
            model = global_patient_risk_model['model']
            label_encoders = global_patient_risk_model['label_encoders']
            
            # Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in input_data:
                    try:
                        input_data[col] = encoder.transform([str(input_data[col])])[0]
                    except ValueError:
                        # Handle unknown categories
                        input_data[col] = 0
        else:
            # Legacy format
            model = global_patient_risk_model
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        predicted_outcome = model.predict(input_df)[0]
        predicted_proba_raw = model.predict_proba(input_df)

        # Get probability for the 'Yes' class
        try:
            yes_class_index = list(model.classes_).index('Yes')
            probability_of_yes = predicted_proba_raw[0, yes_class_index]
        except ValueError:
            # If 'Yes' is not a class, default to 0
            probability_of_yes = 0.0

        return {
            "status": "success",
            "predicted_readmission": predicted_outcome,
            "probability_of_readmission": f"{probability_of_yes:.2f}"
        }

    except Exception as e:
        return {"status": "error", "message": f"Error during risk prediction: {str(e)}"}