from mcp.server.fastmcp import FastMCP, Image
from typing import Optional

# Existing tool imports
from utils.file_reader import add_patient_record, update_patient_record, get_patient_record
from utils.risk import predict_patient_outcome_tool
from utils.compare_efficacy import compare_treatments_tool
from utils.similar import find_similar_patients_tool
from utils.anamoly import detect_data_anomalies_tool

# New tool imports
from utils.analytics import (
    get_record_summary_tool,
    kpi_overview_tool,
    length_of_stay_benchmark_tool,
    satisfaction_drivers_report_tool
)
from utils.los import predict_length_of_stay_tool
from utils.triage import triage_queue_tool
from utils.counterfactual import counterfactual_readmission_tool
from utils.plotting import generate_chart_bytes

# Create the shared MCP server instance
mcp = FastMCP("Hospital Patient Data Management System")

# ============================================================================
# EXISTING TOOLS (unchanged interfaces, risk_assessment now includes SHAP)
# ============================================================================

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
    MCP Tool: Patient Risk Assessment with SHAP Explanations (SQL-based)
    Predicts readmission risk for a patient based on provided attributes,
    and provides SHAP-based explanations for the prediction.

    Args:
        patient_attributes (dict): A dictionary with patient's 'Age', 'Gender',
                                   'Condition', 'Procedure', 'Length_of_Stay'.

    Returns:
        dict: A dictionary with:
            - prediction ('Yes'/'No')
            - probability of readmission
            - SHAP explanation with top feature contributions
            - model metadata
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


# ============================================================================
# NEW ANALYTICS TOOLS
# ============================================================================

@mcp.tool()
def get_record_summary(patient_id: int) -> dict:
    """
    MCP Tool: Get Patient Record Summary with Derived Insights
    Retrieves a patient record with additional derived fields like cost bands,
    LOS bands, and comparison flags against population percentiles.

    Args:
        patient_id (int): The ID of the patient to summarize.

    Returns:
        dict: Patient record with derived insights:
            - record: Original patient data
            - derived: Bands (cost_band, los_band) and flags (is_high_cost, is_long_stay)
            - percentile_thresholds: Population percentiles used for comparison
    """
    return get_record_summary_tool(patient_id)


@mcp.tool()
def kpi_overview(where: Optional[dict] = None) -> dict:
    """
    MCP Tool: Hospital KPI Overview Dashboard
    Provides aggregated key performance indicators for a patient cohort.

    Args:
        where (dict, optional): Filtering criteria (e.g., {"Condition": "Diabetes", "Gender": "Male"})

    Returns:
        dict: Aggregated KPIs including:
            - n: Number of patients
            - avg_cost, median_cost: Cost metrics
            - avg_length_of_stay, median_length_of_stay: LOS metrics
            - readmission_rate: Percentage of readmissions
            - recovery_rate: Percentage of recovered patients
            - avg_satisfaction: Average satisfaction score
    """
    return kpi_overview_tool(where)


@mcp.tool()
def length_of_stay_benchmark(group_by: str = "Condition", where: Optional[dict] = None) -> dict:
    """
    MCP Tool: Length of Stay Benchmarking
    Compares length of stay statistics across different groups (conditions, procedures, etc.)

    Args:
        group_by (str): Column to group by. Options: 'Condition', 'Procedure', 'Gender', 'Outcome', 'Readmission'
        where (dict, optional): Filtering criteria

    Returns:
        dict: LOS benchmarks per group with:
            - n, mean, p50, p75, p90: LOS statistics
            - exceeds_global_p90: Flag if group median exceeds global 90th percentile
            - long_stay_count: Number of long-stay patients in group
    """
    return length_of_stay_benchmark_tool(group_by, where)


@mcp.tool()
def satisfaction_drivers_report(
    group_by: Optional[list] = None,
    min_n: int = 20,
    where: Optional[dict] = None
) -> dict:
    """
    MCP Tool: Patient Satisfaction Drivers Analysis
    Identifies factors associated with low patient satisfaction scores.

    Args:
        group_by (list, optional): Columns to group by for cohort analysis. Default: ["Condition", "Procedure"]
        min_n (int): Minimum cohort size to include in analysis
        where (dict, optional): Filtering criteria

    Returns:
        dict: Satisfaction analysis with:
            - bottom_cohorts: Lowest satisfaction cohorts
            - correlations: Correlation with Cost, LOS, Outcome, Readmission
            - overall_stats: Mean, median, and low satisfaction percentage
    """
    return satisfaction_drivers_report_tool(group_by, min_n, where)


# ============================================================================
# NEW ML TOOLS
# ============================================================================

@mcp.tool()
def predict_length_of_stay(patient_attributes: dict) -> dict:
    """
    MCP Tool: Predict Length of Stay with SHAP Explanations
    Predicts expected hospital stay duration based on patient attributes.

    Args:
        patient_attributes (dict): A dictionary with:
            - Age (int): Patient age
            - Gender (str): Patient gender
            - Condition (str): Medical condition
            - Procedure (str): Medical procedure
            - Cost (float): Treatment cost

    Returns:
        dict: Prediction results with:
            - predicted_length_of_stay_days: Expected LOS
            - prediction_range: 90% confidence interval
            - context: Comparison to population average
            - explanation: SHAP feature contributions
            - model_info: Model version and metrics
    """
    return predict_length_of_stay_tool(patient_attributes)


@mcp.tool()
def triage_queue(
    rules: Optional[list] = None,
    limit: int = 20,
    where: Optional[dict] = None,
    include_risk_probability: bool = False
) -> dict:
    """
    MCP Tool: Generate Patient Triage Priority Queue
    Creates a prioritized list of patients based on configurable rules.

    Args:
        rules (list, optional): List of rule dicts. Each rule has:
            - field: Column name (e.g., 'Cost', 'Length_of_Stay')
            - operator: '>', '>=', '<', '<=', '==', '!='
            - value: Threshold value
            - score: Points if rule matches (default: 1.0)
            - description: Human-readable description
            If None, default rules for high-priority patients are used.
        limit (int): Maximum patients to return (default: 20)
        where (dict, optional): Filtering criteria
        include_risk_probability (bool): Include readmission probability (doctor-only)

    Returns:
        dict: Prioritized patient list with scores and reasons
    """
    return triage_queue_tool(rules, limit, where, include_risk_probability)


@mcp.tool()
def counterfactual_readmission(
    patient_attributes: dict,
    target_probability: float = 0.20
) -> dict:
    """
    MCP Tool: Counterfactual Analysis for Readmission Risk
    Suggests what changes to patient factors could reduce readmission probability.
    
    DISCLAIMER: This is a model-based simulation for analytical purposes only.
    Suggestions are NOT medical advice.

    Args:
        patient_attributes (dict): A dictionary with:
            - Age (int): Patient age
            - Gender (str): Patient gender
            - Condition (str): Medical condition
            - Procedure (str): Medical procedure
            - Length_of_Stay (int): Hospital stay duration
        target_probability (float): Target readmission probability (default: 0.20 = 20%)

    Returns:
        dict: Counterfactual analysis with:
            - current_probability: Current predicted risk
            - target_achievable: Whether target can be reached
            - suggested_changes: List of changes with estimated probabilities
            - disclaimer: Important usage warning
    """
    return counterfactual_readmission_tool(patient_attributes, target_probability)


# ============================================================================
# NEW VISUALIZATION TOOL
# ============================================================================

@mcp.tool()
def generate_chart(
    plot_type: str,
    metric: str,
    group_by: Optional[str] = None,
    time_period: Optional[str] = None,
    where: Optional[dict] = None
):
    """
    MCP Tool: Generate Data Visualization Charts
    Creates various chart types for analyzing hospital data.
    Returns an MCP Image that works in Streamlit, Claude Desktop, and Cursor.

    Args:
        plot_type (str): Chart type. Options:
            - "bar": Bar chart comparing groups
            - "line": Line chart for trends
            - "pie": Pie chart for proportions
            - "histogram": Distribution histogram
            - "boxplot": Box plot for distributions
        metric (str): Metric to visualize. Options:
            - "billing": Total cost
            - "admissions": Number of admissions
            - "age": Patient age
            - "length_of_stay": Hospital stay duration
            - "satisfaction": Satisfaction scores
            - "readmission_rate": Readmission percentage
            - "recovery_rate": Recovery percentage
        group_by (str, optional): Grouping column. Options:
            - "medical_condition" or "condition"
            - "procedure" or "admission_type"
            - "gender"
            - "outcome"
        time_period (str, optional): Date range filter (reserved for future use)
        where (dict, optional): Additional filters

    Returns:
        Image: MCP Image object with PNG data (displayed natively in Claude/Cursor)
        dict: Error information if chart generation fails
    """
    result = generate_chart_bytes(plot_type, metric, group_by, time_period, where)
    
    # If result is bytes, wrap in MCP Image for native display
    if isinstance(result, bytes):
        return Image(data=result, format="png")
    
    # Otherwise return error dict
    return result


def main():
    """Main function to run the MCP server."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()