from mcp.server.fastmcp import FastMCP
from utils.file_reader import add_patient_record, update_patient_record, get_patient_record
from utils.risk import predict_patient_outcome_tool
from utils.compare_efficacy import compare_treatments_tool
from utils.similar import find_similar_patients_tool
from utils.anamoly import detect_data_anomalies_tool

# Create the shared MCP server instance
mcp = FastMCP("Hospital Patient Data Management System")

@mcp.tool()
def add_record(patient_data: dict) -> dict:
    """
    MCP Tool: Ingest Patient Data (SQL-based)
    Adds a new patient record to the database.

    Args:
        patient_data (dict): A dictionary containing all patient attributes including:
                           Age, Gender, Condition, Procedure, Cost, Length_of_Stay, 
                           Readmission, Outcome, Satisfaction

    Returns:
        dict: A dictionary indicating success/failure and the new record ID.
    """
    return add_patient_record(patient_data)

@mcp.tool()
def update_record(patient_id: int, updates: dict) -> dict:
    """
    MCP Tool: Transform/Update Patient Data (SQL-based)
    Updates an existing patient record in the database.

    Args:
        patient_id (int): The ID of the patient record to update.
        updates (dict): A dictionary of fields to update and their new values.

    Returns:
        dict: A dictionary indicating success/failure.
    """
    return update_patient_record(patient_id, updates)

@mcp.tool()
def get_record(patient_id: int = None, query_criteria: dict = None) -> dict:
    """
    MCP Tool: Retrieve Patient Data (SQL-based)
    Retrieves one or more patient records from the database.

    Args:
        patient_id (int, optional): The ID of a specific patient record to retrieve.
        query_criteria (dict, optional): A dictionary of field-value pairs to filter records.

    Returns:
        dict: A dictionary containing the retrieved records or an error message.
    """
    return get_patient_record(patient_id, query_criteria)

@mcp.tool()
def risk_assessment(patient_attributes: dict) -> dict:
    """
    MCP Tool: Patient Risk Assessment (SQL-based)
    Predicts readmission risk for a patient based on provided attributes.

    Args:
        patient_attributes (dict): A dictionary with patient's 'Age', 'Gender',
                                   'Condition', 'Procedure', 'Length_of_Stay'.

    Returns:
        dict: A dictionary with prediction ('Yes'/'No') and probability,
    """
    return predict_patient_outcome_tool(patient_attributes)

@mcp.tool()
def compare_treatment(condition: str, procedures: list[str]) -> dict:
    """
    MCP Tool: Treatment Efficacy Comparison (SQL-based)
    Compares effectiveness metrics for different procedures for a given condition.

    Args:
        condition (str): The medical condition to compare treatments for.
        procedures (list[str]): A list of procedure names to compare.

    Returns:
        dict: A dictionary containing comparison results for each procedure,
              or an error/no data message.
    """
    return compare_treatments_tool(condition, procedures)

@mcp.tool()
def similar_patients(target_patient_attributes: dict, num_similar: int = 5) -> dict:
    """
    MCP Tool: Find Similar Patient Cases (SQL-based)
    Finds historical patient records most similar to the target patient based on key attributes.

    Args:
        target_patient_attributes (dict): A dictionary with key attributes of the patient
                                          to find similar cases for (e.g., 'Age', 'Gender', 'Condition', 'Procedure').
        num_similar (int): The number of most similar patients to return.

    Returns:
        dict: A dictionary containing the most similar patient records,
              or an error/no data message.
    """
    return find_similar_patients_tool(target_patient_attributes, num_similar)

@mcp.tool()
def detect_anomaly(numerical_cols: list[str] = None, categorical_cols: list[str] = None, z_score_threshold: float = 3.0) -> dict:
    """
    MCP Tool: Data Quality & Anomaly Detection (SQL-based)
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
    return detect_data_anomalies_tool(numerical_cols, categorical_cols, z_score_threshold)

def main():
    """Main function to run the MCP server."""
    mcp.run()

if __name__ == "__main__":
    main()