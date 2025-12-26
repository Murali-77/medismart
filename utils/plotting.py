"""
Plotting module for generating hospital data visualizations.
Supports bar, line, pie, histogram, and boxplot chart types.
Returns raw PNG bytes for MCP Image compatibility.
Also saves charts to file for Streamlit client display.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io
import os
import time
from pathlib import Path
from typing import Optional, Union
from .file_reader import load_data_from_db

# Directory for saving charts (for Streamlit client rendering)
CHART_DIR = Path(__file__).parent.parent / "data" / "charts"
LATEST_CHART_PATH = CHART_DIR / "latest_chart.png"
CHART_METADATA_PATH = CHART_DIR / "chart_metadata.txt"


def _ensure_chart_dir():
    """Ensure the charts directory exists."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)


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


def _get_metric_column(metric: str) -> tuple:
    """
    Map metric name to column name and aggregation type.
    
    Returns:
        tuple: (column_name, aggregation_method, display_name)
    """
    metric_mapping = {
        "billing": ("Cost", "sum", "Total Billing ($)"),
        "cost": ("Cost", "mean", "Average Cost ($)"),
        "admissions": ("Patient_ID", "count", "Number of Admissions"),
        "age": ("Age", "mean", "Average Age"),
        "length_of_stay": ("Length_of_Stay", "mean", "Average Length of Stay (days)"),
        "los": ("Length_of_Stay", "mean", "Average Length of Stay (days)"),
        "satisfaction": ("Satisfaction", "mean", "Average Satisfaction Score"),
        "readmission_rate": ("Readmission", "custom", "Readmission Rate (%)"),
        "recovery_rate": ("Outcome", "custom", "Recovery Rate (%)")
    }
    
    return metric_mapping.get(metric.lower(), (None, None, None))


def _get_group_column(group_by: str) -> str:
    """Map group_by name to actual column name."""
    group_mapping = {
        "medical_condition": "Condition",
        "condition": "Condition",
        "insurance_provider": "Condition",  # Dataset doesn't have insurance, use Condition
        "admission_type": "Procedure",  # Map to Procedure as closest match
        "procedure": "Procedure",
        "gender": "Gender",
        "outcome": "Outcome",
        "readmission": "Readmission"
    }
    
    return group_mapping.get(group_by.lower(), group_by)


def _create_figure():
    """Create a matplotlib figure with consistent styling."""
    # Use smaller figure size to reduce image file size for LLM context limits
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    return fig, ax


def _fig_to_bytes(fig, chart_title: str = "Chart") -> bytes:
    """
    Convert matplotlib figure to PNG bytes with compression.
    Also saves to file for Streamlit client rendering.
    """
    buf = io.BytesIO()
    # Use lower dpi for smaller file size (keeps under LLM token limits)
    # PNG with lower dpi is more efficient than JPEG for charts
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    
    # Save to file for Streamlit client to display
    _ensure_chart_dir()
    with open(LATEST_CHART_PATH, 'wb') as f:
        f.write(img_bytes)
    
    # Save metadata (timestamp and title) for the Streamlit client
    with open(CHART_METADATA_PATH, 'w') as f:
        f.write(f"{time.time()}\n{chart_title}")
    
    plt.close(fig)
    return img_bytes


def _create_bar_chart(data: pd.DataFrame, group_col: str, value_col: str, 
                      agg_method: str, title: str, ylabel: str) -> bytes:
    """Create a bar chart and return PNG bytes."""
    if agg_method == "custom":
        if "Readmission" in value_col:
            grouped = data.groupby(group_col).apply(
                lambda x: (x['Readmission'] == 'Yes').sum() / len(x) * 100
            ).reset_index(name='value')
        elif "Outcome" in value_col:
            grouped = data.groupby(group_col).apply(
                lambda x: (x['Outcome'] == 'Recovered').sum() / len(x) * 100
            ).reset_index(name='value')
        else:
            grouped = data.groupby(group_col)[value_col].mean().reset_index(name='value')
    else:
        grouped = data.groupby(group_col)[value_col].agg(agg_method).reset_index(name='value')
    
    grouped = grouped.sort_values('value', ascending=False).head(15)
    
    fig, ax = _create_figure()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grouped)))
    
    bars = ax.bar(range(len(grouped)), grouped['value'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, grouped['value']):
        height = bar.get_height()
        ax.annotate(f'{val:,.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_bytes(fig, title)


def _create_line_chart(data: pd.DataFrame, group_col: str, value_col: str,
                       agg_method: str, title: str, ylabel: str) -> bytes:
    """Create a line chart and return PNG bytes."""
    if agg_method == "custom":
        if "Readmission" in value_col:
            grouped = data.groupby(group_col).apply(
                lambda x: (x['Readmission'] == 'Yes').sum() / len(x) * 100
            ).reset_index(name='value')
        else:
            grouped = data.groupby(group_col)[value_col].mean().reset_index(name='value')
    else:
        grouped = data.groupby(group_col)[value_col].agg(agg_method).reset_index(name='value')
    
    fig, ax = _create_figure()
    
    ax.plot(range(len(grouped)), grouped['value'], marker='o', linewidth=2, 
            markersize=8, color='#2196F3', markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(range(len(grouped)), grouped['value'], alpha=0.1, color='#2196F3')
    
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return _fig_to_bytes(fig, title)


def _create_pie_chart(data: pd.DataFrame, group_col: str, value_col: str,
                      agg_method: str, title: str) -> bytes:
    """Create a pie chart and return PNG bytes."""
    if agg_method == "count" or value_col == "Patient_ID":
        grouped = data.groupby(group_col).size().reset_index(name='value')
    else:
        grouped = data.groupby(group_col)[value_col].agg(agg_method).reset_index(name='value')
    
    grouped = grouped.nlargest(8, 'value')  # Top 8 for readability
    
    fig, ax = _create_figure()
    colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
    
    wedges, texts, autotexts = ax.pie(
        grouped['value'], 
        labels=grouped[group_col],
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(grouped),
        shadow=True,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return _fig_to_bytes(fig, title)


def _create_histogram(data: pd.DataFrame, value_col: str, title: str, xlabel: str) -> bytes:
    """Create a histogram and return PNG bytes."""
    fig, ax = _create_figure()
    
    values = data[value_col].dropna()
    
    n, bins, patches = ax.hist(values, bins=20, color='#4CAF50', edgecolor='white', 
                                linewidth=0.5, alpha=0.8)
    
    # Add mean and median lines
    mean_val = values.mean()
    median_val = values.median()
    ax.axvline(mean_val, color='#F44336', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='#FF9800', linestyle='-.', linewidth=2, label=f'Median: {median_val:.1f}')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return _fig_to_bytes(fig, title)


def _create_boxplot(data: pd.DataFrame, group_col: str, value_col: str, 
                    title: str, ylabel: str) -> bytes:
    """Create a boxplot and return PNG bytes."""
    fig, ax = _create_figure()
    
    groups = data[group_col].unique()[:10]  # Limit to 10 groups
    plot_data = [data[data[group_col] == g][value_col].dropna() for g in groups]
    
    bp = ax.boxplot(plot_data, patch_artist=True, labels=groups)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
    
    for median in bp['medians']:
        median.set_color('#F44336')
        median.set_linewidth(2)
    
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return _fig_to_bytes(fig, title)


def generate_chart_bytes(
    plot_type: str,
    metric: str,
    group_by: Optional[str] = None,
    time_period: Optional[str] = None,
    where: Optional[dict] = None
) -> Union[bytes, dict]:
    """
    Generate a chart and return PNG bytes or error dict.
    
    This function is used by the MCP tool to generate charts.
    Returns raw PNG bytes on success, or a dict with error details on failure.
    
    Args:
        plot_type: Type of chart ("bar", "line", "pie", "histogram", "boxplot")
        metric: Metric to visualize
        group_by: Column to group data by
        time_period: Optional date range filter (reserved for future use)
        where: Optional additional filters
    
    Returns:
        bytes: PNG image data on success
        dict: Error information on failure
    """
    # Validate plot_type
    valid_plot_types = ['bar', 'line', 'pie', 'histogram', 'boxplot']
    if plot_type.lower() not in valid_plot_types:
        return {
            "status": "error",
            "message": f"Invalid plot_type. Valid options: {valid_plot_types}"
        }
    
    # Get metric mapping
    column, agg_method, display_name = _get_metric_column(metric)
    if column is None:
        valid_metrics = ['billing', 'cost', 'admissions', 'age', 'length_of_stay', 
                         'los', 'satisfaction', 'readmission_rate', 'recovery_rate']
        return {
            "status": "error",
            "message": f"Invalid metric. Valid options: {valid_metrics}"
        }
    
    # Load data
    df = load_data_from_db()
    if df.empty:
        return {"status": "error", "message": "Database is empty."}
    
    # Apply filters
    filtered_df = _apply_where_filter(df, where)
    if filtered_df.empty:
        return {"status": "no_data", "message": "No records match the specified criteria."}
    
    # Map group_by to actual column
    if group_by:
        group_col = _get_group_column(group_by)
        if group_col not in filtered_df.columns:
            return {
                "status": "error",
                "message": f"Group column '{group_by}' not found. Available: {list(filtered_df.columns)}"
            }
    else:
        # Default grouping based on plot type
        group_col = "Condition" if plot_type.lower() in ['bar', 'line', 'pie', 'boxplot'] else None
    
    try:
        plot_type_lower = plot_type.lower()
        title = f"{display_name} by {group_col}" if group_col else display_name
        
        if plot_type_lower == 'bar':
            if not group_col:
                return {"status": "error", "message": "Bar chart requires group_by parameter."}
            return _create_bar_chart(filtered_df, group_col, column, agg_method, title, display_name)
        
        elif plot_type_lower == 'line':
            if not group_col:
                return {"status": "error", "message": "Line chart requires group_by parameter."}
            return _create_line_chart(filtered_df, group_col, column, agg_method, title, display_name)
        
        elif plot_type_lower == 'pie':
            if not group_col:
                group_col = "Condition"
            title = f"Distribution of {display_name} by {group_col}"
            return _create_pie_chart(filtered_df, group_col, column, agg_method, title)
        
        elif plot_type_lower == 'histogram':
            if column not in filtered_df.columns:
                return {"status": "error", "message": f"Column '{column}' not found for histogram."}
            title = f"Distribution of {display_name}"
            return _create_histogram(filtered_df, column, title, display_name)
        
        elif plot_type_lower == 'boxplot':
            if not group_col:
                return {"status": "error", "message": "Boxplot requires group_by parameter."}
            if column not in filtered_df.columns:
                return {"status": "error", "message": f"Column '{column}' not found for boxplot."}
            return _create_boxplot(filtered_df, group_col, column, title, display_name)
        
        else:
            return {"status": "error", "message": f"Unsupported plot type: {plot_type}"}
        
    except Exception as e:
        return {"status": "error", "message": f"Error generating chart: {str(e)}"}


def get_latest_chart() -> Optional[tuple]:
    """
    Get the latest generated chart for Streamlit client display.
    
    Returns:
        tuple: (image_bytes, title, timestamp) or None if no chart exists
    """
    if not LATEST_CHART_PATH.exists() or not CHART_METADATA_PATH.exists():
        return None
    
    try:
        with open(CHART_METADATA_PATH, 'r') as f:
            lines = f.read().strip().split('\n')
            timestamp = float(lines[0])
            title = lines[1] if len(lines) > 1 else "Chart"
        
        with open(LATEST_CHART_PATH, 'rb') as f:
            image_bytes = f.read()
        
        return (image_bytes, title, timestamp)
    except Exception:
        return None


def clear_latest_chart():
    """Clear the latest chart file after displaying."""
    try:
        if LATEST_CHART_PATH.exists():
            os.remove(LATEST_CHART_PATH)
        if CHART_METADATA_PATH.exists():
            os.remove(CHART_METADATA_PATH)
    except Exception:
        pass


def get_available_chart_options() -> dict:
    """
    Get available options for chart generation.
    
    Returns:
        dict: Available plot types, metrics, and grouping options
    """
    return {
        "status": "success",
        "plot_types": ["bar", "line", "pie", "histogram", "boxplot"],
        "metrics": {
            "billing": "Total billing/cost amount",
            "cost": "Average cost per case",
            "admissions": "Number of patient admissions",
            "age": "Patient age",
            "length_of_stay": "Hospital stay duration (days)",
            "satisfaction": "Patient satisfaction scores (1-5)",
            "readmission_rate": "Percentage of readmissions",
            "recovery_rate": "Percentage of recovered patients"
        },
        "group_by_options": {
            "medical_condition": "Group by medical condition",
            "condition": "Group by medical condition",
            "procedure": "Group by medical procedure",
            "admission_type": "Group by procedure/admission type",
            "gender": "Group by patient gender",
            "outcome": "Group by treatment outcome"
        },
        "example_request": {
            "plot_type": "bar",
            "metric": "billing",
            "group_by": "medical_condition",
            "time_period": None,
            "where": {"Gender": "Female"}
        }
    }
