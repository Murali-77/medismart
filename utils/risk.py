"""
Risk assessment module with SHAP explanations for readmission prediction.
Model must be trained separately using train_risk_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Correct path to model file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / 'patient_risk_model.joblib'


def _to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif pd.isna(val):
        return None
    return val


def _load_model():
    """Loads the pre-trained model. Returns None if model not found."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)
    else:
        print(f"Model not found at {MODEL_FILE}. Please train the model first using: python utils/train_risk_model.py")
        return None


def _compute_shap_explanation(model, input_df, feature_columns, label_encoders, original_values):
    """
    Compute SHAP explanations for the prediction.
    
    Args:
        model: Trained model (RandomForestClassifier)
        input_df: Encoded input DataFrame
        feature_columns: List of feature column names
        label_encoders: Dict of label encoders for categorical features
        original_values: Original (unencoded) patient attributes
    
    Returns:
        dict: SHAP explanation with top features and their contributions
    """
    try:
        import shap
        
        # Create TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # For binary classification, shap_values is a list [class_0, class_1]
        # We want the SHAP values for the "Yes" (readmission) class
        try:
            yes_class_index = list(model.classes_).index('Yes')
        except ValueError:
            yes_class_index = 1  # Default to second class
        
        if isinstance(shap_values, list):
            # SHAP version returns list for classifier
            shap_for_yes = shap_values[yes_class_index][0]
        else:
            # Newer SHAP versions may return different structure
            if len(shap_values.shape) == 3:
                shap_for_yes = shap_values[0, :, yes_class_index]
            else:
                shap_for_yes = shap_values[0]
        
        # Build feature explanations
        feature_explanations = []
        for i, col in enumerate(feature_columns):
            shap_val = shap_for_yes[i]
            
            # Get original value for display
            if col in original_values:
                original_val = original_values[col]
            else:
                original_val = input_df[col].iloc[0]
            
            feature_explanations.append({
                "feature": col,
                "value": _to_python_type(original_val),
                "shap_value": _to_python_type(round(float(shap_val), 4)),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "abs_importance": _to_python_type(round(abs(float(shap_val)), 4))
            })
        
        # Sort by absolute importance (most important first)
        feature_explanations.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        # Get expected value (base value)
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = explainer.expected_value[yes_class_index]
        else:
            base_value = explainer.expected_value
        
        return {
            "status": "available",
            "base_value": _to_python_type(round(float(base_value), 4)),
            "top_features": feature_explanations[:5],  # Top 5 features
            "all_features": feature_explanations
        }
        
    except ImportError:
        return {
            "status": "unavailable",
            "error": "SHAP library not installed. Run: pip install shap"
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": f"SHAP computation failed: {str(e)}"
        }


# Load the model once when the script starts
global_patient_risk_model = _load_model()


def predict_patient_outcome_tool(patient_attributes: dict) -> dict:
    """
    MCP Tool: Patient Risk Assessment with SHAP Explanations (SQL-based)
    Predicts readmission risk for a patient based on provided attributes,
    and provides SHAP-based explanations for the prediction.

    Args:
        patient_attributes (dict): A dictionary with patient's 'Age', 'Gender',
                                   'Condition', 'Procedure', 'Length_of_Stay'.

    Returns:
        dict: A dictionary with prediction ('Yes'/'No'), probability,
              SHAP explanation, and model metadata.
    """
    if global_patient_risk_model is None:
        return {
            "status": "error", 
            "message": "Prediction model not available. Please train it first using: python utils/train_risk_model.py"
        }

    # Ensure all required attributes are present
    required_attrs = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
    if not all(attr in patient_attributes for attr in required_attrs):
        return {"status": "error", "message": f"Missing required patient attributes. Needed: {required_attrs}"}

    try:
        # Store original values for explanation display
        original_values = patient_attributes.copy()
        
        # Prepare input data
        input_data = patient_attributes.copy()
        
        # If model_data is a dict (new format with label encoders)
        if isinstance(global_patient_risk_model, dict):
            model = global_patient_risk_model['model']
            label_encoders = global_patient_risk_model['label_encoders']
            feature_columns = global_patient_risk_model.get('feature_columns', required_attrs)
            
            # Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in input_data:
                    try:
                        input_data[col] = encoder.transform([str(input_data[col])])[0]
                    except ValueError:
                        # Handle unknown categories
                        input_data[col] = 0
        else:
            # Legacy format (Pipeline model from train_risk_model.py)
            model = global_patient_risk_model
            label_encoders = {}
            feature_columns = required_attrs
        
        # Ensure correct column order
        input_data_ordered = {col: input_data[col] for col in feature_columns}
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data_ordered])
        
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

        # Compute SHAP explanation (only for dict format models)
        if isinstance(global_patient_risk_model, dict):
            explanation = _compute_shap_explanation(
                model, input_df, feature_columns, label_encoders, original_values
            )
        else:
            explanation = {
                "status": "unavailable",
                "error": "SHAP not available for Pipeline models. Retrain with train_risk_model.py for SHAP support."
            }
        
        # Build response
        result = {
            "status": "success",
            "predicted_readmission": _to_python_type(predicted_outcome),
            "probability_of_readmission": _to_python_type(round(float(probability_of_yes), 4)),
            "probability_of_readmission_pct": f"{probability_of_yes * 100:.1f}%",
            "explanation": explanation
        }
        
        # Add model metadata if available
        if isinstance(global_patient_risk_model, dict):
            result["model_info"] = {
                "version": global_patient_risk_model.get('model_version', 'unknown'),
                "trained_on_n": global_patient_risk_model.get('trained_on_n', 'unknown')
            }
        
        return result

    except Exception as e:
        return {"status": "error", "message": f"Error during risk prediction: {str(e)}"}


def get_model_info() -> dict:
    """
    Get information about the current readmission prediction model.
    
    Returns:
        dict: Model metadata including version, training size, and accuracy.
    """
    if global_patient_risk_model is None:
        return {"status": "error", "message": "Model not loaded."}
    
    if isinstance(global_patient_risk_model, dict):
        return {
            "status": "success",
            "model_type": "RandomForestClassifier",
            "version": global_patient_risk_model.get('model_version', 'unknown'),
            "trained_on_n": global_patient_risk_model.get('trained_on_n', 'unknown'),
            "train_accuracy": _to_python_type(global_patient_risk_model.get('train_accuracy')),
            "test_accuracy": _to_python_type(global_patient_risk_model.get('test_accuracy')),
            "features": global_patient_risk_model.get('feature_columns', [])
        }
    else:
        return {
            "status": "success",
            "model_type": "Pipeline (Logistic Regression)",
            "version": "legacy",
            "message": "Legacy model format - limited metadata available"
        }
