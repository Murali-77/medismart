import pandas as pd
from pathlib import Path
import os

# Base directory where our data lives
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
file_path = DATA_DIR / "hospital-data-analysis.csv"

def load_data_from_csv():
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError: return pd.DataFrame()
    return pd.DataFrame()

def _save_data_to_csv(df):
    """Helper function to save DataFrame to the CSV file."""
    df.to_csv(file_path, index=False)

def add_patient_record(patient_data: dict):

    df=load_data_from_csv()

    required_fields = ['Age', 'Gender', 'Condition', 'Procedure', 'Cost','Length_of_Stay', 'Readmission', 'Outcome', 'Satisfaction']
    if not all(field in patient_data for field in required_fields):
        return {"status": "error", "message": f"Missing required fields. Required: {required_fields}"}

    try:
        # Generate a new Patient_ID
        if df.empty or 'Patient_ID' not in df.columns:
            new_id = 1
        else:
            # Ensure Patient_ID is numeric for max calculation, handle non-numeric gracefully
            df['Patient_ID_numeric'] = pd.to_numeric(df['Patient_ID'], errors='coerce')
            max_id = df['Patient_ID_numeric'].max()
            new_id = int(max_id) + 1 if pd.notna(max_id) else 1
            df = df.drop(columns=['Patient_ID_numeric']) # Clean up temp column

        patient_data['Patient_ID'] = new_id

        # Convert new patient data to a DataFrame row
        new_df_row = pd.DataFrame([patient_data])

        # Append new record and save
        df = pd.concat([df, new_df_row], ignore_index=True)
        _save_data_to_csv(df)

        return {"status": "success", "message": "Patient record added successfully to CSV", "patient_id": new_id}

    except Exception as e:
        return {"status": "error", "message": f"Failed to add patient record to CSV: {str(e)}"}
    
def update_patient_record_csv(patient_id: int, updates: dict) -> dict:
    """
    MCP Tool: Transform/Update Patient Data (CSV-based)
    Updates an existing patient record in the CSV file.

    Args:
        patient_id (int): The ID of the patient record to update.
        updates (dict): A dictionary of fields to update and their new values.

    Returns:
        dict: A dictionary indicating success/failure.
    """
    if not patient_id:
        return {"status": "error", "message": "Patient ID is required for update."}
    if not updates:
        return {"status": "error", "message": "No updates provided."}

    df = load_data_from_csv()

    if df.empty:
        return {"status": "error", "message": "CSV file is empty or not found."}

    try:
        # Find the row(s) to update
        # Ensure Patient_ID column is treated as numeric for comparison
        df['Patient_ID_numeric'] = pd.to_numeric(df['Patient_ID'], errors='coerce')
        idx_to_update = df[df['Patient_ID_numeric'] == patient_id].index

        if not idx_to_update.empty:
            for col, value in updates.items():
                if col in df.columns: # Only update if column exists
                    df.loc[idx_to_update, col] = value
                else:
                    print(f"Warning: Column '{col}' not found in CSV, skipping update for this field.")

            df = df.drop(columns=['Patient_ID_numeric']) # Clean up temp column
            _save_data_to_csv(df)
            return {"status": "success", "message": f"Patient record {patient_id} updated successfully in CSV."}
        else:
            df = df.drop(columns=['Patient_ID_numeric']) # Clean up temp column
            return {"status": "error", "message": f"Patient record with ID {patient_id} not found in CSV."}

    except Exception as e:
        return {"status": "error", "message": f"Failed to update patient record in CSV: {str(e)}"}


def get_patient_record_csv(patient_id: int = None, query_criteria: dict = None) -> dict:
    
    if not patient_id and not query_criteria:
        return {"status": "error", "message": "Either patient_id or query_criteria must be provided."}

    df = load_data_from_csv()

    if df.empty:
        return {"status": "no_records", "message": "CSV file is empty or not found."}

    try:
        results_df = pd.DataFrame()
        if patient_id:
            df['Patient_ID_numeric'] = pd.to_numeric(df['Patient_ID'], errors='coerce')
            results_df = df[df['Patient_ID_numeric'] == patient_id].drop(columns=['Patient_ID_numeric'])
        elif query_criteria:
            query_mask = pd.Series([True] * len(df))
            for field, value in query_criteria.items():
                if field in df.columns:
                    query_mask &= (df[field] == value)
                else:
                    return {"status": "error", "message": f"Query field '{field}' not found in CSV columns."}
            results_df = df[query_mask]

        if not results_df.empty:
            return {"status": "success", "records": results_df.to_dict(orient='records')}
        else:
            return {"status": "no_records", "message": "No patient records found matching the criteria in CSV."}

    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve patient record(s) from CSV: {str(e)}"}