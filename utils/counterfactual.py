"""
Counterfactual analysis module for readmission risk reduction.
Suggests actionable changes to reduce predicted readmission probability.
"""
import pandas as pd
import numpy as np
from typing import Optional
from .file_reader import load_data_from_db


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


def _get_procedure_options_for_condition(condition: str) -> list:
    """
    Get available procedures for a given condition from historical data.
    
    Args:
        condition: Medical condition name
    
    Returns:
        list: Procedures used for this condition in the dataset
    """
    df = load_data_from_db()
    if df.empty or 'Condition' not in df.columns or 'Procedure' not in df.columns:
        return []
    
    condition_df = df[df['Condition'].str.lower() == condition.lower()]
    if condition_df.empty:
        return []
    
    # Get procedures with their success metrics
    procedures = condition_df.groupby('Procedure').agg({
        'Patient_ID': 'count',
        'Readmission': lambda x: (x == 'Yes').mean()
    }).reset_index()
    procedures.columns = ['Procedure', 'count', 'readmission_rate']
    
    # Filter to procedures with reasonable sample size
    procedures = procedures[procedures['count'] >= 5]
    
    # Sort by readmission rate (lower is better)
    procedures = procedures.sort_values('readmission_rate')
    
    return procedures['Procedure'].tolist()


def counterfactual_readmission_tool(
    patient_attributes: dict,
    target_probability: float = 0.20
) -> dict:
    """
    MCP Tool: Counterfactual Analysis for Readmission Risk Reduction
    
    Suggests what changes to patient attributes could reduce the predicted
    readmission probability to a target level.
    
    IMPORTANT DISCLAIMER: This is a model-based simulation tool for educational
    and analytical purposes only. Suggestions are NOT medical advice and should
    not be used to make clinical decisions without proper medical consultation.
    
    Args:
        patient_attributes: Dict with patient's attributes:
            - 'Age' (int): Patient age
            - 'Gender' (str): Patient gender
            - 'Condition' (str): Medical condition
            - 'Procedure' (str): Medical procedure
            - 'Length_of_Stay' (int): Hospital stay duration
        target_probability: Target readmission probability (default: 0.20 = 20%)
    
    Returns:
        dict: Counterfactual analysis with suggested changes and their effects
    """
    # Validate inputs
    required_attrs = ['Age', 'Gender', 'Condition', 'Procedure', 'Length_of_Stay']
    missing = [attr for attr in required_attrs if attr not in patient_attributes]
    if missing:
        return {"status": "error", "message": f"Missing required patient attributes: {missing}"}
    
    if not 0 < target_probability < 1:
        return {"status": "error", "message": "target_probability must be between 0 and 1"}
    
    try:
        # Import risk assessment tool
        from .risk import predict_patient_outcome_tool, global_patient_risk_model
        
        if global_patient_risk_model is None:
            return {"status": "error", "message": "Risk model not available."}
        
        # Get current prediction
        current_result = predict_patient_outcome_tool(patient_attributes)
        if current_result.get('status') != 'success':
            return {"status": "error", "message": f"Failed to get current prediction: {current_result.get('message')}"}
        
        current_probability = float(current_result.get('probability_of_readmission', 0))
        
        # If already below target, no changes needed
        if current_probability <= target_probability:
            return {
                "status": "success",
                "message": "Patient already at or below target probability.",
                "current_probability": _to_python_type(round(current_probability, 4)),
                "current_probability_pct": f"{current_probability * 100:.1f}%",
                "target_probability": _to_python_type(target_probability),
                "target_probability_pct": f"{target_probability * 100:.1f}%",
                "suggested_changes": [],
                "disclaimer": "This is a model-based simulation, not medical advice."
            }
        
        # Define actionable changes to explore
        # We focus on operationally adjustable factors
        suggested_changes = []
        
        original_los = patient_attributes['Length_of_Stay']
        original_procedure = patient_attributes['Procedure']
        condition = patient_attributes['Condition']
        
        # 1. Explore Length of Stay changes (±1 to ±5 days within bounds)
        los_candidates = []
        for delta in range(-3, 4):
            if delta == 0:
                continue
            new_los = original_los + delta
            if 1 <= new_los <= 30:  # Reasonable bounds
                los_candidates.append(new_los)
        
        for new_los in los_candidates:
            test_attrs = patient_attributes.copy()
            test_attrs['Length_of_Stay'] = new_los
            
            test_result = predict_patient_outcome_tool(test_attrs)
            if test_result.get('status') == 'success':
                new_prob = float(test_result.get('probability_of_readmission', current_probability))
                
                if new_prob < current_probability:
                    suggested_changes.append({
                        "change_type": "length_of_stay",
                        "field": "Length_of_Stay",
                        "from": _to_python_type(original_los),
                        "to": _to_python_type(new_los),
                        "change_description": f"{'Increase' if new_los > original_los else 'Decrease'} LOS by {abs(new_los - original_los)} day(s)",
                        "current_probability": _to_python_type(round(current_probability, 4)),
                        "estimated_probability": _to_python_type(round(new_prob, 4)),
                        "probability_reduction": _to_python_type(round(current_probability - new_prob, 4)),
                        "probability_reduction_pct": f"{(current_probability - new_prob) * 100:.1f}%",
                        "meets_target": new_prob <= target_probability
                    })
        
        # 2. Explore alternative procedures for the same condition
        alternative_procedures = _get_procedure_options_for_condition(condition)
        
        for alt_proc in alternative_procedures:
            if alt_proc == original_procedure:
                continue
            
            test_attrs = patient_attributes.copy()
            test_attrs['Procedure'] = alt_proc
            
            test_result = predict_patient_outcome_tool(test_attrs)
            if test_result.get('status') == 'success':
                new_prob = float(test_result.get('probability_of_readmission', current_probability))
                
                if new_prob < current_probability:
                    suggested_changes.append({
                        "change_type": "procedure",
                        "field": "Procedure",
                        "from": original_procedure,
                        "to": alt_proc,
                        "change_description": f"Alternative procedure: {alt_proc}",
                        "current_probability": _to_python_type(round(current_probability, 4)),
                        "estimated_probability": _to_python_type(round(new_prob, 4)),
                        "probability_reduction": _to_python_type(round(current_probability - new_prob, 4)),
                        "probability_reduction_pct": f"{(current_probability - new_prob) * 100:.1f}%",
                        "meets_target": new_prob <= target_probability,
                        "note": "Based on historical data for similar patients. Clinical assessment required."
                    })
        
        # 3. Explore combined changes (LOS + Procedure)
        if len(alternative_procedures) > 0 and len(los_candidates) > 0:
            # Try best procedure with optimal LOS
            best_alt_proc = alternative_procedures[0] if alternative_procedures[0] != original_procedure else (
                alternative_procedures[1] if len(alternative_procedures) > 1 else None
            )
            
            if best_alt_proc:
                for new_los in los_candidates[:3]:  # Top 3 LOS options
                    test_attrs = patient_attributes.copy()
                    test_attrs['Length_of_Stay'] = new_los
                    test_attrs['Procedure'] = best_alt_proc
                    
                    test_result = predict_patient_outcome_tool(test_attrs)
                    if test_result.get('status') == 'success':
                        new_prob = float(test_result.get('probability_of_readmission', current_probability))
                        
                        if new_prob < current_probability * 0.9:  # At least 10% improvement
                            suggested_changes.append({
                                "change_type": "combined",
                                "changes": [
                                    {"field": "Length_of_Stay", "from": original_los, "to": new_los},
                                    {"field": "Procedure", "from": original_procedure, "to": best_alt_proc}
                                ],
                                "change_description": f"Combined: LOS to {new_los} days + {best_alt_proc}",
                                "current_probability": _to_python_type(round(current_probability, 4)),
                                "estimated_probability": _to_python_type(round(new_prob, 4)),
                                "probability_reduction": _to_python_type(round(current_probability - new_prob, 4)),
                                "probability_reduction_pct": f"{(current_probability - new_prob) * 100:.1f}%",
                                "meets_target": new_prob <= target_probability
                            })
        
        # Sort by probability reduction (best first)
        suggested_changes.sort(key=lambda x: x.get('probability_reduction', 0), reverse=True)
        
        # Keep top suggestions
        top_suggestions = suggested_changes[:10]
        
        # Find if any meets target
        target_met = any(s.get('meets_target', False) for s in top_suggestions)
        
        return {
            "status": "success",
            "current_probability": _to_python_type(round(current_probability, 4)),
            "current_probability_pct": f"{current_probability * 100:.1f}%",
            "target_probability": _to_python_type(target_probability),
            "target_probability_pct": f"{target_probability * 100:.1f}%",
            "target_achievable": target_met,
            "total_options_explored": len(suggested_changes),
            "suggested_changes": top_suggestions,
            "input_attributes": {k: _to_python_type(v) for k, v in patient_attributes.items()},
            "disclaimer": (
                "⚠️ IMPORTANT: This is a data-driven simulation based on historical patterns. "
                "These suggestions are NOT medical advice and should not be used to make "
                "clinical decisions without proper medical consultation and assessment."
            )
        }
        
    except ImportError as e:
        return {"status": "error", "message": f"Required module not available: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Error during counterfactual analysis: {str(e)}"}

