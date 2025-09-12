import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .file_reader import load_data_from_db

def find_similar_patients_tool(target_patient_attributes: dict, num_similar: int = 5) -> dict:
    """
    MCP Tool: Find Similar Patient Cases (SQL-based)
    Finds historical patient records most similar to the target patient based on key attributes.

    Args:
        target_patient_attributes (dict): A dictionary with key attributes of the patient
                                          to find similar cases for (e.g., 'Age', 'Gender', 'Condition').
        num_similar (int): The number of most similar patients to return.

    Returns:
        dict: A dictionary containing the most similar patient records,
              or an error/no data message.
    """
    df = load_data_from_db()

    if df.empty:
        return {"status": "error", "message": "Database is empty."}

    # Define features to use for similarity (can be customized)
    # These should ideally be the same features used in your prediction models for consistency
    similarity_features = ['Age', 'Gender', 'Condition', 'Procedure']

    # Filter out records where essential similarity features might be missing
    df_cleaned = df.dropna(subset=similarity_features)

    if df_cleaned.empty:
        return {"status": "no_data", "message": "No valid data rows for similarity comparison after cleaning."}

    # Prepare target patient data
    target_df = pd.DataFrame([target_patient_attributes])
    target_df = target_df[similarity_features] # Ensure target has same columns

    # Combine for consistent one-hot encoding and scaling
    combined_df = pd.concat([df_cleaned[similarity_features], target_df], ignore_index=True)

    # One-hot encode categorical features (Gender, Condition, Procedure)
    categorical_features = [f for f in ['Gender', 'Condition', 'Procedure'] if f in combined_df.columns]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(combined_df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # Identify numerical features and scale them
    numerical_features = [f for f in ['Age', 'Length_of_Stay'] if f in combined_df.columns]
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(combined_df[numerical_features])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features)

    # Recombine all processed features
    processed_combined_df = pd.concat([scaled_numerical_df, encoded_df], axis=1)

    # Separate original data and target patient data after processing
    processed_original_df = processed_combined_df.iloc[:-1]
    processed_target_patient = processed_combined_df.iloc[-1:]

    # Calculate Euclidean distance (or other distance metric)
    # Reshape for broadcast if target_patient is 1 row
    distances = np.linalg.norm(processed_original_df.values - processed_target_patient.values, axis=1)

    # Add distances back to the original (cleaned) DataFrame
    df_cleaned['distance'] = distances

    # Sort by distance and get top N similar patients
    similar_patients_df = df_cleaned.sort_values(by='distance').head(num_similar)

    # Remove distance column and convert to list of dicts for output
    similar_patients_list = similar_patients_df.drop(columns=['distance']).to_dict(orient='records')

    if not similar_patients_list:
        return {"status": "no_records", "message": "No similar patient records found."}

    return {"status": "success", "similar_patients": similar_patients_list}