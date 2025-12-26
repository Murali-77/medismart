"""
Triage queue module for prioritizing patient follow-up.
Supports rule-based scoring with optional risk model integration.
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


def _apply_where_filter(df: pd.DataFrame, where: Optional[dict]) -> pd.DataFrame:
    """Apply filtering criteria to DataFrame."""
    if not where or df.empty:
        return df
    
    filtered_df = df.copy()
    for col, val in where.items():
        if col in df.columns:
            filtered_df = filtered_df[filtered_df[col] == val]
    
    return filtered_df


def _evaluate_rule(row: pd.Series, rule: dict) -> tuple:
    """
    Evaluate a single rule against a patient row.
    
    Args:
        row: Patient record as pandas Series
        rule: Rule dict with 'field', 'operator', 'value', and optional 'score'
    
    Returns:
        tuple: (matched: bool, score: float, reason: str)
    """
    field = rule.get('field')
    operator = rule.get('operator', '>')
    threshold = rule.get('value')
    score = rule.get('score', 1.0)
    
    if field not in row.index or pd.isna(row[field]):
        return (False, 0, None)
    
    patient_value = row[field]
    matched = False
    
    try:
        if operator == '>':
            matched = patient_value > threshold
        elif operator == '>=':
            matched = patient_value >= threshold
        elif operator == '<':
            matched = patient_value < threshold
        elif operator == '<=':
            matched = patient_value <= threshold
        elif operator == '==':
            matched = patient_value == threshold
        elif operator == '!=':
            matched = patient_value != threshold
        elif operator == 'in':
            matched = patient_value in threshold
    except Exception:
        return (False, 0, None)
    
    if matched:
        reason = f"{field} {operator} {threshold} (actual: {_to_python_type(patient_value)})"
        return (True, score, reason)
    
    return (False, 0, None)


def _get_default_rules() -> list:
    """Return default triage rules for common priority scenarios."""
    return [
        {
            "field": "Length_of_Stay",
            "operator": ">",
            "value": 7,
            "score": 2.0,
            "description": "Extended stay (>7 days)"
        },
        {
            "field": "Cost",
            "operator": ">",
            "value": 25000,
            "score": 1.5,
            "description": "High cost case"
        },
        {
            "field": "Satisfaction",
            "operator": "<=",
            "value": 2,
            "score": 2.5,
            "description": "Low satisfaction"
        },
        {
            "field": "Readmission",
            "operator": "==",
            "value": "Yes",
            "score": 3.0,
            "description": "Previous readmission"
        },
        {
            "field": "Outcome",
            "operator": "==",
            "value": "Stable",
            "score": 1.0,
            "description": "Non-recovered outcome"
        }
    ]


def triage_queue_tool(
    rules: Optional[list] = None,
    limit: int = 20,
    where: Optional[dict] = None,
    include_risk_probability: bool = False
) -> dict:
    """
    MCP Tool: Generate prioritized patient triage queue.
    
    Produces a prioritized list of patients based on configurable rules.
    Rules can reference patient fields and assign scores.
    
    Args:
        rules: List of rule dicts. Each rule has:
            - 'field': Column name (e.g., 'Cost', 'Length_of_Stay', 'Satisfaction')
            - 'operator': '>', '>=', '<', '<=', '==', '!=', 'in'
            - 'value': Threshold value
            - 'score': Points to add if rule matches (default: 1.0)
            - 'description': Optional human-readable description
            If None, default rules are used.
        limit: Maximum number of patients to return (default: 20)
        where: Optional filtering criteria (e.g., {"Condition": "Diabetes"})
        include_risk_probability: If True and user has doctor role, include
            readmission probability from risk model (default: False)
    
    Returns:
        dict: Prioritized patient list with scores and matched reasons
    """
    df = load_data_from_db()
    
    if df.empty:
        return {"status": "error", "message": "Database is empty."}
    
    # Apply filters
    filtered_df = _apply_where_filter(df, where)
    
    if filtered_df.empty:
        return {"status": "no_data", "message": "No records match the specified criteria."}
    
    # Use default rules if none provided
    if rules is None:
        rules = _get_default_rules()
    
    # Validate rules
    if not isinstance(rules, list):
        return {"status": "error", "message": "Rules must be a list of rule dictionaries."}
    
    # Compute scores for each patient
    patient_scores = []
    
    for idx, row in filtered_df.iterrows():
        total_score = 0.0
        matched_reasons = []
        
        for rule in rules:
            matched, score, reason = _evaluate_rule(row, rule)
            if matched:
                total_score += score
                desc = rule.get('description', reason)
                matched_reasons.append(desc)
        
        if total_score > 0:
            patient_entry = {
                "Patient_ID": _to_python_type(row.get('Patient_ID')),
                "score": _to_python_type(round(total_score, 2)),
                "reasons": matched_reasons,
                "summary": {
                    "Age": _to_python_type(row.get('Age')),
                    "Gender": _to_python_type(row.get('Gender')),
                    "Condition": _to_python_type(row.get('Condition')),
                    "Procedure": _to_python_type(row.get('Procedure')),
                    "Length_of_Stay": _to_python_type(row.get('Length_of_Stay')),
                    "Cost": _to_python_type(row.get('Cost')),
                    "Satisfaction": _to_python_type(row.get('Satisfaction')),
                    "Outcome": _to_python_type(row.get('Outcome')),
                    "Readmission": _to_python_type(row.get('Readmission'))
                }
            }
            
            # Optionally add risk probability (doctor-only feature)
            if include_risk_probability:
                try:
                    from .risk import predict_patient_outcome_tool
                    risk_input = {
                        'Age': row.get('Age'),
                        'Gender': row.get('Gender'),
                        'Condition': row.get('Condition'),
                        'Procedure': row.get('Procedure'),
                        'Length_of_Stay': row.get('Length_of_Stay')
                    }
                    risk_result = predict_patient_outcome_tool(risk_input)
                    if risk_result.get('status') == 'success':
                        patient_entry['readmission_probability'] = risk_result.get('probability_of_readmission')
                        # Add to score based on high risk
                        prob = float(risk_result.get('probability_of_readmission', 0))
                        if prob > 0.5:
                            patient_entry['score'] += 2.0
                            patient_entry['reasons'].append(f"High readmission risk ({prob:.0%})")
                except Exception:
                    pass  # Skip risk integration if it fails
            
            patient_scores.append(patient_entry)
    
    # Sort by score descending
    patient_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Limit results
    top_patients = patient_scores[:limit]
    
    return {
        "status": "success",
        "total_flagged": len(patient_scores),
        "returned": len(top_patients),
        "filters_applied": where if where else "none",
        "rules_applied": len(rules),
        "patients": top_patients
    }


def get_available_triage_fields() -> dict:
    """
    Get list of fields available for triage rules.
    
    Returns:
        dict: Available fields with their types and suggested operators
    """
    return {
        "status": "success",
        "fields": {
            "Age": {"type": "numeric", "operators": [">", ">=", "<", "<="]},
            "Cost": {"type": "numeric", "operators": [">", ">=", "<", "<="]},
            "Length_of_Stay": {"type": "numeric", "operators": [">", ">=", "<", "<="]},
            "Satisfaction": {"type": "numeric", "operators": [">", ">=", "<", "<=", "=="]},
            "Gender": {"type": "categorical", "operators": ["==", "!=", "in"]},
            "Condition": {"type": "categorical", "operators": ["==", "!=", "in"]},
            "Procedure": {"type": "categorical", "operators": ["==", "!=", "in"]},
            "Outcome": {"type": "categorical", "operators": ["==", "!="], "values": ["Recovered", "Stable"]},
            "Readmission": {"type": "categorical", "operators": ["==", "!="], "values": ["Yes", "No"]}
        },
        "example_rule": {
            "field": "Cost",
            "operator": ">",
            "value": 20000,
            "score": 1.5,
            "description": "High cost case"
        }
    }

