"""
Length of Stay (LOS) prediction module with SHAP explanations.
Model must be trained separately using train_los_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Path to model file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / 'patient_los_model.joblib'


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
    """Loads the pre-trained LOS model. Returns None if model not found."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading LOS model from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)
    else:
        print(f"LOS model not found at {MODEL_FILE}. Please train the model first using: python utils/train_los_model.py")
        return None


def _compute_shap_explanation(model, input_df, feature_columns, original_values):
    """
    Compute SHAP explanations for LOS prediction.
    
    Args:
        model: Trained RandomForestRegressor
        input_df: Encoded input DataFrame
        feature_columns: List of feature column names
        original_values: Original (unencoded) patient attributes
    
    Returns:
        dict: SHAP explanation with feature contributions
    """
    try:
        import shap
        
        # Create TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values (for regressor, returns single array)
        shap_values = explainer.shap_values(input_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Build feature explanations
        feature_explanations = []
        for i, col in enumerate(feature_columns):
            shap_val = shap_vals[i] if isinstance(shap_vals, np.ndarray) else shap_vals
            
            # Get original value for display
            original_val = original_values.get(col, input_df[col].iloc[0])
            
            feature_explanations.append({
                "feature": col,
                "value": _to_python_type(original_val),
                "shap_value": _to_python_type(round(float(shap_val), 4)),
                "direction": "increases_los" if shap_val > 0 else "decreases_los",
                "abs_importance": _to_python_type(round(abs(float(shap_val)), 4)),
                "contribution_days": _to_python_type(round(float(shap_val), 2))
            })
        
        # Sort by absolute importance
        feature_explanations.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        # Get expected value (average LOS in training data)
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        return {
            "status": "available",
            "base_value_days": _to_python_type(round(float(base_value), 2)),
            "interpretation": f"Base expected LOS is {round(float(base_value), 1)} days, adjusted by feature contributions",
            "top_features": feature_explanations[:5],
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
global_los_model = _load_model()


def predict_length_of_stay_tool(patient_attributes: dict) -> dict:
    """
    MCP Tool: Predict Length of Stay with SHAP Explanations
    Predicts hospital stay duration based on patient attributes.

    Args:
        patient_attributes (dict): A dictionary with patient's attributes:
            - 'Age' (int): Patient age
            - 'Gender' (str): Patient gender
            - 'Condition' (str): Medical condition
            - 'Procedure' (str): Medical procedure
            - 'Cost' (float): Treatment cost

    Returns:
        dict: A dictionary with predicted LOS, confidence interval,
              SHAP explanation, and model metadata.
    """
    if global_los_model is None:
        return {
            "status": "error", 
            "message": "LOS prediction model not available. Please train it first using: python utils/train_los_model.py"
        }

    # Required attributes (Cost included as per user request)
    required_attrs = ['Age', 'Gender', 'Condition', 'Procedure', 'Cost']
    missing = [attr for attr in required_attrs if attr not in patient_attributes]
    if missing:
        return {"status": "error", "message": f"Missing required patient attributes: {missing}"}

    try:
        # Store original values for explanation display
        original_values = patient_attributes.copy()
        
        # Prepare input data
        input_data = patient_attributes.copy()
        
        # Get model components
        model = global_los_model['model']
        label_encoders = global_los_model['label_encoders']
        feature_columns = global_los_model['feature_columns']
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data:
                try:
                    input_data[col] = encoder.transform([str(input_data[col])])[0]
                except ValueError:
                    # Handle unknown categories - use most frequent
                    input_data[col] = 0
        
        # Ensure correct column order
        input_data_ordered = {col: input_data[col] for col in feature_columns}
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data_ordered])
        
        # Make prediction
        predicted_los = model.predict(input_df)[0]
        
        # Get prediction interval using tree variance
        # Each tree in the forest gives a prediction
        tree_predictions = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
        prediction_std = tree_predictions.std()
        
        # 90% confidence interval
        ci_lower = max(1, predicted_los - 1.645 * prediction_std)
        ci_upper = predicted_los + 1.645 * prediction_std
        
        # Compute SHAP explanation
        explanation = _compute_shap_explanation(
            model, input_df, feature_columns, original_values
        )
        
        # Get target stats for context
        target_stats = global_los_model.get('target_stats', {})
        
        # Build response
        result = {
            "status": "success",
            "predicted_length_of_stay_days": _to_python_type(round(float(predicted_los), 1)),
            "prediction_range": {
                "lower_bound_days": _to_python_type(round(float(ci_lower), 1)),
                "upper_bound_days": _to_python_type(round(float(ci_upper), 1)),
                "confidence_level": "90%"
            },
            "context": {
                "population_mean_days": _to_python_type(round(target_stats.get('mean', 0), 1)),
                "population_std_days": _to_python_type(round(target_stats.get('std', 0), 1)),
                "vs_population": "above_average" if predicted_los > target_stats.get('mean', 0) else "below_average"
            },
            "explanation": explanation,
            "model_info": {
                "version": global_los_model.get('model_version', 'unknown'),
                "trained_on_n": global_los_model.get('trained_on_n', 'unknown'),
                "test_mae_days": _to_python_type(round(global_los_model.get('metrics', {}).get('test_mae', 0), 2))
            }
        }
        
        return result

    except Exception as e:
        return {"status": "error", "message": f"Error during LOS prediction: {str(e)}"}


def get_los_model_info() -> dict:
    """
    Get information about the current LOS prediction model.
    
    Returns:
        dict: Model metadata including version, training size, and metrics.
    """
    if global_los_model is None:
        return {"status": "error", "message": "LOS model not loaded."}
    
    metrics = global_los_model.get('metrics', {})
    
    return {
        "status": "success",
        "model_type": "RandomForestRegressor",
        "version": global_los_model.get('model_version', 'unknown'),
        "trained_on_n": global_los_model.get('trained_on_n', 'unknown'),
        "features": global_los_model.get('feature_columns', []),
        "metrics": {
            "train_mae": _to_python_type(round(metrics.get('train_mae', 0), 2)),
            "test_mae": _to_python_type(round(metrics.get('test_mae', 0), 2)),
            "train_rmse": _to_python_type(round(metrics.get('train_rmse', 0), 2)),
            "test_rmse": _to_python_type(round(metrics.get('test_rmse', 0), 2)),
            "train_r2": _to_python_type(round(metrics.get('train_r2', 0), 3)),
            "test_r2": _to_python_type(round(metrics.get('test_r2', 0), 3))
        },
        "target_stats": global_los_model.get('target_stats', {})
    }
