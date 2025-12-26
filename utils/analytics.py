"""
Analytics utilities for hospital data MCP tools.
Provides KPI overview, benchmarking, patient summaries, and satisfaction analysis.
"""
import pandas as pd
import numpy as np
from typing import Optional
from .file_reader import load_data_from_db, get_patient_record


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
    """
    Apply filtering criteria to DataFrame.
    Mirrors get_patient_record semantics with field whitelist.
    """
    if not where or df.empty:
        return df
    
    valid_columns = set(df.columns)
    filtered_df = df.copy()
    
    for col, val in where.items():
        if col in valid_columns:
            filtered_df = filtered_df[filtered_df[col] == val]
    
    return filtered_df


def _compute_percentile_bands(df: pd.DataFrame) -> dict:
    """Compute percentile cutoffs for Cost and Length_of_Stay."""
    bands = {}
    
    if 'Cost' in df.columns and not df['Cost'].dropna().empty:
        cost_series = df['Cost'].dropna()
        bands['cost_p25'] = _to_python_type(cost_series.quantile(0.25))
        bands['cost_p50'] = _to_python_type(cost_series.quantile(0.50))
        bands['cost_p75'] = _to_python_type(cost_series.quantile(0.75))
        bands['cost_p90'] = _to_python_type(cost_series.quantile(0.90))
    
    if 'Length_of_Stay' in df.columns and not df['Length_of_Stay'].dropna().empty:
        los_series = df['Length_of_Stay'].dropna()
        bands['los_p25'] = _to_python_type(los_series.quantile(0.25))
        bands['los_p50'] = _to_python_type(los_series.quantile(0.50))
        bands['los_p75'] = _to_python_type(los_series.quantile(0.75))
        bands['los_p90'] = _to_python_type(los_series.quantile(0.90))
    
    return bands


def get_record_summary_tool(patient_id: int) -> dict:
    """
    Get a comprehensive summary of a patient record with derived fields.
    
    Args:
        patient_id: The ID of the patient to summarize.
    
    Returns:
        dict: Patient record with derived bands and flags.
    """
    # Get the patient record
    result = get_patient_record(patient_id=patient_id)
    
    if result.get("status") != "success":
        return result
    
    records = result.get("records", [])
    if not records:
        return {"status": "no_records", "message": f"No patient found with ID {patient_id}"}
    
    patient = dict(records[0])
    
    # Load full dataset to compute percentile bands
    df = load_data_from_db()
    if df.empty:
        return {"status": "success", "record": patient, "derived": {}, "message": "No historical data for comparison"}
    
    bands = _compute_percentile_bands(df)
    
    # Derive bands and flags for this patient
    derived = {}
    
    # Cost band
    if 'Cost' in patient and patient['Cost'] is not None:
        cost = patient['Cost']
        if cost <= bands.get('cost_p25', float('inf')):
            derived['cost_band'] = 'low'
        elif cost <= bands.get('cost_p50', float('inf')):
            derived['cost_band'] = 'below_median'
        elif cost <= bands.get('cost_p75', float('inf')):
            derived['cost_band'] = 'above_median'
        elif cost <= bands.get('cost_p90', float('inf')):
            derived['cost_band'] = 'high'
        else:
            derived['cost_band'] = 'very_high'
        derived['is_high_cost'] = cost > bands.get('cost_p75', float('inf'))
    
    # LOS band
    if 'Length_of_Stay' in patient and patient['Length_of_Stay'] is not None:
        los = patient['Length_of_Stay']
        if los <= bands.get('los_p25', float('inf')):
            derived['los_band'] = 'short'
        elif los <= bands.get('los_p50', float('inf')):
            derived['los_band'] = 'below_median'
        elif los <= bands.get('los_p75', float('inf')):
            derived['los_band'] = 'above_median'
        elif los <= bands.get('los_p90', float('inf')):
            derived['los_band'] = 'long'
        else:
            derived['los_band'] = 'very_long'
        derived['is_long_stay'] = los > bands.get('los_p75', float('inf'))
    
    # Satisfaction flag
    if 'Satisfaction' in patient and patient['Satisfaction'] is not None:
        derived['is_low_satisfaction'] = patient['Satisfaction'] <= 2
    
    # Convert patient record values to Python types
    patient = {k: _to_python_type(v) for k, v in patient.items()}
    
    return {
        "status": "success",
        "record": patient,
        "derived": derived,
        "percentile_thresholds": bands
    }


def kpi_overview_tool(where: Optional[dict] = None) -> dict:
    """
    Get aggregated KPIs for a cohort of patients.
    
    Args:
        where: Optional filtering criteria (e.g., {"Condition": "Diabetes"}).
    
    Returns:
        dict: Aggregated KPIs including costs, LOS, rates, and satisfaction.
    """
    df = load_data_from_db()
    
    if df.empty:
        return {"status": "error", "message": "Database is empty."}
    
    # Apply filters
    filtered_df = _apply_where_filter(df, where)
    
    if filtered_df.empty:
        return {"status": "no_data", "message": "No records match the specified criteria."}
    
    n = len(filtered_df)
    
    kpis = {
        "n": n,
        "filters_applied": where if where else "none"
    }
    
    # Cost metrics
    if 'Cost' in filtered_df.columns:
        cost_series = filtered_df['Cost'].dropna()
        if not cost_series.empty:
            kpis['avg_cost'] = _to_python_type(round(cost_series.mean(), 2))
            kpis['median_cost'] = _to_python_type(round(cost_series.median(), 2))
            kpis['min_cost'] = _to_python_type(round(cost_series.min(), 2))
            kpis['max_cost'] = _to_python_type(round(cost_series.max(), 2))
    
    # Length of Stay metrics
    if 'Length_of_Stay' in filtered_df.columns:
        los_series = filtered_df['Length_of_Stay'].dropna()
        if not los_series.empty:
            kpis['avg_length_of_stay'] = _to_python_type(round(los_series.mean(), 2))
            kpis['median_length_of_stay'] = _to_python_type(round(los_series.median(), 2))
    
    # Readmission rate
    if 'Readmission' in filtered_df.columns:
        readmission_series = filtered_df['Readmission'].dropna()
        if not readmission_series.empty:
            readmission_count = (readmission_series == 'Yes').sum()
            kpis['readmission_rate'] = _to_python_type(round(readmission_count / len(readmission_series) * 100, 2))
            kpis['readmission_rate_pct'] = f"{kpis['readmission_rate']}%"
    
    # Recovery rate
    if 'Outcome' in filtered_df.columns:
        outcome_series = filtered_df['Outcome'].dropna()
        if not outcome_series.empty:
            recovered_count = (outcome_series == 'Recovered').sum()
            kpis['recovery_rate'] = _to_python_type(round(recovered_count / len(outcome_series) * 100, 2))
            kpis['recovery_rate_pct'] = f"{kpis['recovery_rate']}%"
    
    # Satisfaction metrics
    if 'Satisfaction' in filtered_df.columns:
        sat_series = filtered_df['Satisfaction'].dropna()
        if not sat_series.empty:
            kpis['avg_satisfaction'] = _to_python_type(round(sat_series.mean(), 2))
            kpis['low_satisfaction_count'] = _to_python_type((sat_series <= 2).sum())
    
    return {"status": "success", "kpis": kpis}


def length_of_stay_benchmark_tool(group_by: str = "Condition", where: Optional[dict] = None) -> dict:
    """
    Benchmark Length of Stay statistics grouped by a categorical field.
    
    Args:
        group_by: Column to group by (default: "Condition").
        where: Optional filtering criteria.
    
    Returns:
        dict: LOS benchmarks per group with percentiles and long-stay flags.
    """
    df = load_data_from_db()
    
    if df.empty:
        return {"status": "error", "message": "Database is empty."}
    
    # Validate group_by column
    valid_group_columns = ['Condition', 'Procedure', 'Gender', 'Outcome', 'Readmission']
    if group_by not in valid_group_columns:
        return {"status": "error", "message": f"Invalid group_by column. Valid options: {valid_group_columns}"}
    
    if group_by not in df.columns:
        return {"status": "error", "message": f"Column '{group_by}' not found in data."}
    
    # Apply filters
    filtered_df = _apply_where_filter(df, where)
    
    if filtered_df.empty:
        return {"status": "no_data", "message": "No records match the specified criteria."}
    
    if 'Length_of_Stay' not in filtered_df.columns:
        return {"status": "error", "message": "Length_of_Stay column not found."}
    
    # Compute global p90 for long-stay threshold
    global_p90 = filtered_df['Length_of_Stay'].quantile(0.90)
    
    # Group and compute stats
    benchmarks = []
    grouped = filtered_df.groupby(group_by)['Length_of_Stay']
    
    for group_name, group_data in grouped:
        if len(group_data) == 0:
            continue
        
        stats = {
            "group": _to_python_type(group_name),
            "n": len(group_data),
            "mean": _to_python_type(round(group_data.mean(), 2)),
            "p50": _to_python_type(round(group_data.quantile(0.50), 2)),
            "p75": _to_python_type(round(group_data.quantile(0.75), 2)),
            "p90": _to_python_type(round(group_data.quantile(0.90), 2)),
            "min": _to_python_type(group_data.min()),
            "max": _to_python_type(group_data.max()),
        }
        
        # Flag if group's median exceeds global p90
        stats["exceeds_global_p90"] = stats["p50"] > global_p90
        stats["long_stay_count"] = _to_python_type((group_data > global_p90).sum())
        
        benchmarks.append(stats)
    
    # Sort by mean LOS descending
    benchmarks.sort(key=lambda x: x['mean'], reverse=True)
    
    return {
        "status": "success",
        "group_by": group_by,
        "filters_applied": where if where else "none",
        "global_p90_threshold": _to_python_type(round(global_p90, 2)),
        "benchmarks": benchmarks
    }


def satisfaction_drivers_report_tool(
    group_by: Optional[list] = None,
    min_n: int = 20,
    where: Optional[dict] = None
) -> dict:
    """
    Analyze satisfaction drivers by identifying low-performing cohorts and correlations.
    
    Args:
        group_by: Columns to group by for cohort analysis (default: ["Condition", "Procedure"]).
        min_n: Minimum cohort size to include in analysis.
        where: Optional filtering criteria.
    
    Returns:
        dict: Satisfaction analysis with bottom cohorts and correlations.
    """
    if group_by is None:
        group_by = ["Condition", "Procedure"]
    
    df = load_data_from_db()
    
    if df.empty:
        return {"status": "error", "message": "Database is empty."}
    
    if 'Satisfaction' not in df.columns:
        return {"status": "error", "message": "Satisfaction column not found."}
    
    # Validate group_by columns
    valid_columns = ['Condition', 'Procedure', 'Gender', 'Outcome', 'Readmission']
    invalid_cols = [c for c in group_by if c not in valid_columns]
    if invalid_cols:
        return {"status": "error", "message": f"Invalid group_by columns: {invalid_cols}. Valid: {valid_columns}"}
    
    # Apply filters
    filtered_df = _apply_where_filter(df, where)
    
    if filtered_df.empty:
        return {"status": "no_data", "message": "No records match the specified criteria."}
    
    result = {
        "status": "success",
        "filters_applied": where if where else "none",
        "min_cohort_size": min_n,
    }
    
    # 1. Bottom cohorts by average satisfaction
    existing_group_cols = [c for c in group_by if c in filtered_df.columns]
    if existing_group_cols:
        grouped = filtered_df.groupby(existing_group_cols).agg({
            'Satisfaction': ['mean', 'count', 'std'],
            'Patient_ID': 'count'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = existing_group_cols + ['avg_satisfaction', 'satisfaction_count', 'satisfaction_std', 'n']
        
        # Filter by min_n
        grouped = grouped[grouped['n'] >= min_n]
        
        if not grouped.empty:
            # Sort by average satisfaction ascending (worst first)
            grouped = grouped.sort_values('avg_satisfaction')
            
            bottom_cohorts = []
            for _, row in grouped.head(10).iterrows():
                cohort = {col: _to_python_type(row[col]) for col in existing_group_cols}
                cohort['avg_satisfaction'] = _to_python_type(round(row['avg_satisfaction'], 2))
                cohort['n'] = _to_python_type(row['n'])
                cohort['std'] = _to_python_type(round(row['satisfaction_std'], 2)) if pd.notna(row['satisfaction_std']) else None
                bottom_cohorts.append(cohort)
            
            result['bottom_cohorts'] = bottom_cohorts
        else:
            result['bottom_cohorts'] = []
            result['bottom_cohorts_message'] = f"No cohorts with n >= {min_n}"
    
    # 2. Correlations with numeric/binary variables
    correlations = {}
    
    # Numeric correlations
    numeric_cols = ['Cost', 'Length_of_Stay', 'Age']
    for col in numeric_cols:
        if col in filtered_df.columns:
            valid_data = filtered_df[[col, 'Satisfaction']].dropna()
            if len(valid_data) > 10:
                corr = valid_data[col].corr(valid_data['Satisfaction'])
                correlations[col] = {
                    "correlation": _to_python_type(round(corr, 3)),
                    "interpretation": "negative" if corr < -0.1 else ("positive" if corr > 0.1 else "weak/none")
                }
    
    # Binary correlations (Outcome, Readmission)
    if 'Outcome' in filtered_df.columns:
        outcome_sat = filtered_df.groupby('Outcome')['Satisfaction'].mean()
        if 'Recovered' in outcome_sat.index and 'Stable' in outcome_sat.index:
            correlations['Outcome'] = {
                "recovered_avg": _to_python_type(round(outcome_sat.get('Recovered', 0), 2)),
                "stable_avg": _to_python_type(round(outcome_sat.get('Stable', 0), 2)),
                "interpretation": "recovery associated with higher satisfaction" if outcome_sat.get('Recovered', 0) > outcome_sat.get('Stable', 0) else "no clear pattern"
            }
    
    if 'Readmission' in filtered_df.columns:
        readmit_sat = filtered_df.groupby('Readmission')['Satisfaction'].mean()
        if 'Yes' in readmit_sat.index and 'No' in readmit_sat.index:
            correlations['Readmission'] = {
                "readmitted_avg": _to_python_type(round(readmit_sat.get('Yes', 0), 2)),
                "not_readmitted_avg": _to_python_type(round(readmit_sat.get('No', 0), 2)),
                "interpretation": "readmission associated with lower satisfaction" if readmit_sat.get('Yes', 0) < readmit_sat.get('No', 0) else "no clear pattern"
            }
    
    result['correlations'] = correlations
    
    # 3. Overall satisfaction stats
    sat_series = filtered_df['Satisfaction'].dropna()
    result['overall_stats'] = {
        "mean": _to_python_type(round(sat_series.mean(), 2)),
        "median": _to_python_type(round(sat_series.median(), 2)),
        "std": _to_python_type(round(sat_series.std(), 2)),
        "low_satisfaction_pct": _to_python_type(round((sat_series <= 2).sum() / len(sat_series) * 100, 2))
    }
    
    return result

