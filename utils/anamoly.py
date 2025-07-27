import pandas as pd
import numpy as np
from scipy.stats import zscore
from .file_reader import load_data_from_csv

def detect_data_anomalies_tool_csv(numerical_cols: list[str] = None, categorical_cols: list[str] = None, z_score_threshold: float = 3.0) -> dict:
    """
    MCP Tool: Data Quality & Anomaly Detection (CSV-based)
    Detects anomalies in numerical and specified categorical columns.

    Args:
        numerical_cols (list[str], optional): List of numerical columns to check for outliers (Z-score).
                                              Defaults to ['Age', 'Cost', 'Length_of_Stay', 'Satisfaction'].
        categorical_cols (list[str], optional): List of categorical columns to check for low-frequency combinations.
                                                Defaults to ['Condition', 'Procedure', 'Outcome', 'Readmission'].
        z_score_threshold (float): Z-score threshold for numerical outlier detection.

    Returns:
        dict: A dictionary containing detected anomalies, or a success message if none.
    """
    df = load_data_from_csv()

    if df.empty:
        return {"status": "error", "message": "CSV file is empty or not found."}

    anomalies = {"numerical_outliers": [], "low_frequency_combinations": []}

    # Default columns if not provided
    if numerical_cols is None:
        numerical_cols = ['Age', 'Cost', 'Length_of_Stay', 'Satisfaction']
    if categorical_cols is None:
        categorical_cols = ['Condition', 'Procedure', 'Outcome', 'Readmission']

    # 1. Numerical Outlier Detection (using Z-score)
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Calculate Z-scores, handling NaNs
            z_scores = np.abs(zscore(df[col].dropna()))
            outliers_indices = z_scores > z_score_threshold
            # Map back to original dataframe indices
            original_indices = df[col].dropna().index[outliers_indices]

            for idx in original_indices:
                anomalies["numerical_outliers"].append({
                    "patient_id": df.loc[idx, 'Patient_ID'],
                    "column": col,
                    "value": df.loc[idx, col],
                    "reason": f"Outlier (Z-score > {z_score_threshold})"
                })
        else:
            print(f"Warning: Numerical column '{col}' not found or not numeric.")

    # 2. Low Frequency Categorical Combinations
    # Focus on potentially critical combinations
    combination_cols = [col for col in ['Condition', 'Procedure', 'Outcome'] if col in df.columns]
    if len(combination_cols) >= 2: # Need at least two columns to form a combination
        # Consider combinations of Condition, Procedure, Outcome
        combinations = df[combination_cols].value_counts(normalize=True)
        # Define a low-frequency threshold (e.g., less than 0.5% of total data)
        low_freq_threshold = 0.005 # 0.5%

        for combo, freq in combinations.items():
            if freq < low_freq_threshold:
                # Find the actual rows that match this low-frequency combination
                # Construct a boolean mask for the combination
                mask = True
                for i, col in enumerate(combination_cols):
                    mask &= (df[col] == combo[i])

                for idx in df[mask].index:
                    anomalies["low_frequency_combinations"].append({
                        "patient_id": df.loc[idx, 'Patient_ID'],
                        "combination": {k: v for k, v in zip(combination_cols, combo)},
                        "frequency": f"{freq:.4f}",
                        "reason": f"Very low frequency combination (< {low_freq_threshold*100:.2f}%)"
                    })
    else:
        print("Warning: Not enough categorical columns for combination analysis or columns not found.")


    if not anomalies["numerical_outliers"] and not anomalies["low_frequency_combinations"]:
        return {"status": "success", "message": "No significant anomalies detected based on current thresholds."}
    else:
        return {"status": "anomalies_found", "anomalies": anomalies}