import pandas as pd
import os
from .file_reader import load_data_from_csv

def compare_treatments_tool_csv(condition: str, procedures: list[str]) -> dict:
    """
    MCP Tool: Treatment Efficacy Comparison (CSV-based)
    Compares effectiveness metrics for different procedures for a given condition.

    Args:
        condition (str): The medical condition to compare treatments for.
        procedures (list[str]): A list of procedure names to compare.

    Returns:
        dict: A dictionary containing comparison results for each procedure,
              or an error/no data message.
    """
    df = load_data_from_csv()

    if df.empty:
        return {"status": "error", "message": "CSV file is empty or not found."}

    # Filter data for the specified condition
    condition_df = df[df['Condition'].str.contains(condition, case=False, na=False)]

    if condition_df.empty:
        return {"status": "no_data", "message": f"No data found for condition: '{condition}'."}

    results = {}
    for proc in procedures:
        # Filter for the specific procedure within the condition
        proc_df = condition_df[condition_df['Procedure'].str.contains(proc, case=False, na=False)]

        if not proc_df.empty:
            # Calculate metrics
            total_cases = len(proc_df)
            recovered_cases = (proc_df['Outcome'] == 'Recovered').sum()
            success_rate = (recovered_cases / total_cases) * 100 if total_cases > 0 else 0
            avg_length_of_stay = proc_df['Length_of_Stay'].mean()
            avg_cost = proc_df['Cost'].mean()

            results[proc] = {
                "total_cases": int(total_cases),
                "success_rate": f"{success_rate:.2f}%",
                "average_length_of_stay_days": f"{avg_length_of_stay:.2f}",
                "average_cost": f"${avg_cost:.2f}"
            }
        else:
            results[proc] = {"message": "No data found for this procedure under the specified condition."}

    if not results:
        return {"status": "error", "message": "No valid procedures provided or data found for them."}

    return {"status": "success", "comparison_results": results}