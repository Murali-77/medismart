# Hospital MCP Chatbot - Testing Queries Guide

This document provides comprehensive test queries for all available MCP tools, including single-tool and multi-tool workflow examples.

---

## 📋 Data Management Tools

### `get_record` - Retrieve Patient Records

```text
# By Patient ID
Get patient record with ID 5

# With filters
Show me all female patients with diabetes

# Multiple criteria
Find patients over 60 with heart disease

# Specific fields
Get the name, age, and condition for patient 15
```

### `add_record` - Add New Patient Records

```text
Add a new patient record: 45 year old male with diabetes, underwent insulin therapy, cost $8500, stayed 4 days, no readmission, recovered, satisfaction 4

Create a patient record for a 62 year old female with heart disease who had angioplasty, cost was $25000, 7 day stay, readmitted, stable outcome, satisfaction 3
```

### `update_record` - Update Existing Records

```text
Update patient 5's satisfaction score to 5

Change patient 10's outcome to "Recovered"

Update the cost for patient 15 to $12000
```

### `get_record_summary` - Patient Record Summary

```text
Give me a summary of patient 5's record

Summarize all information for patient 12

What's the overview of patient 25?
```

---

## 🔬 Analytics Tools

### `kpi_overview` - Key Performance Indicators

```text
Show me the hospital KPIs

What are the current key performance indicators?

Give me an overview of hospital metrics

Display KPI dashboard data
```

### `length_of_stay_benchmark` - LOS Benchmarking

```text
Show length of stay benchmarks by condition

What's the average stay duration for different medical conditions?

Compare length of stay across procedures

How does our LOS compare across patient conditions?
```

### `satisfaction_drivers_report` - Patient Satisfaction Analysis

```text
What are the main drivers of patient satisfaction?

Generate a satisfaction drivers report

Analyze what factors affect patient satisfaction scores

Show me the satisfaction analysis report
```

### `detect_anomaly` - Data Quality & Outliers

```text
Check for anomalies in patient data

Detect any outliers in our records

Are there any data quality issues?

Find unusual patterns in the patient database
```

---

## 🤖 ML Prediction Tools

### `risk_assessment` - Readmission Risk Prediction

```text
# With full patient details
Assess readmission risk for a 65 year old male with diabetes who had insulin therapy and stayed 5 days

# Predict risk
What's the readmission probability for a 55 year old female with heart disease, had angioplasty, 4 day stay?

# With SHAP explanations
Explain the risk factors for a 70 year old male with stroke, underwent rehabilitation, 10 day stay
```

### `predict_length_of_stay` - LOS Prediction

```text
Predict length of stay for a 50 year old male with heart disease undergoing angioplasty, expected cost $20000

How long will a 45 year old female with cancer likely stay after chemotherapy?

Estimate hospital stay duration for a 60 year old diabetic patient needing insulin therapy
```

### `counterfactual_readmission` - What-If Analysis

```text
# With patient details
Run counterfactual analysis: what if a 65 year old male diabetic with 5 day stay had different treatment?

# Analyze intervention options
What changes would reduce readmission risk for a 70 year old heart disease patient?

Show me counterfactual scenarios for a stroke patient aged 68 with 8 day stay
```

---

## 🏥 Clinical Tools

### `compare_treatment` - Treatment Efficacy Comparison

```text
Compare angioplasty vs cardiac catheterization for heart disease

Which treatment is more effective for diabetes: insulin therapy or medication management?

Compare outcomes between surgery and chemotherapy for cancer patients

Analyze treatment effectiveness for stroke: rehabilitation vs surgery
```

### `similar_patients` - Find Similar Cases

```text
Find patients similar to a 55 year old male with diabetes

Show me similar cases to patient 10

Find historical cases matching a 65 year old female with heart disease

Search for patients with similar profile to ID 25
```

### `triage_queue` - Patient Prioritization

```text
Show me the current triage queue

Generate a priority list of patients needing attention

Which patients should be seen first based on urgency?

Display the triage priority queue with risk scores
```

---

## 📊 Visualization Tool

### `generate_chart` - Data Visualization

```text
# Bar Charts
Generate a bar chart for age grouped by gender
Create a bar chart showing admissions by medical condition
Show billing amounts by procedure as a bar chart

# Line Charts
Create a line chart of average age by condition
Show length of stay trends by procedure

# Pie Charts
Generate a pie chart of patient distribution by condition
Show a pie chart of admissions by gender

# Histograms
Create a histogram of patient ages
Show the distribution of length of stay
Generate a histogram of billing costs

# Box Plots
Create a boxplot of age by condition
Show satisfaction scores distribution by outcome
```

---

## 🔄 Multi-Tool Workflow Queries

These queries demonstrate the ReAct agent's ability to chain multiple tools together.

### Patient Lookup + Risk Assessment

```text
Get patient 5's record and assess their readmission risk

Fetch patient 10's details and predict if they'll be readmitted

Show me patient 15 and then calculate their risk score with explanations
```

### Patient Lookup + Counterfactual Analysis

```text
Get patient 5's record and run a counterfactual analysis on them

Fetch patient 12's data and show me what-if scenarios for reducing their readmission risk

Look up patient 20 and analyze alternative treatment outcomes
```

### Patient Lookup + LOS Prediction

```text
Get patient 8's record and predict their expected length of stay

Fetch patient 25's details and estimate how long they should stay

Show me patient 30 and predict their hospital stay duration
```

### Patient Lookup + Similar Patients

```text
Get patient 5's record and find similar historical cases

Fetch patient 10's details and search for patients with matching profiles

Look up patient 15 and show me comparable cases from our database
```

### KPI Analysis + Visualization

```text
Show me the KPIs and then generate a bar chart of admissions by condition

Get the hospital metrics and create a pie chart of patient distribution

Display KPI overview and visualize the satisfaction trends
```

### Multiple Patient Comparison

```text
Get records for patients 5, 10, and 15, then compare their treatment outcomes

Fetch patient 8 and patient 12, and compare their conditions and costs

Look up patients 20 and 25 and analyze their readmission patterns
```

### Complex Multi-Step Workflows

```text
Get patient 5's record, assess their risk, and then run counterfactual analysis

Show me patient 10's details, predict their length of stay, and find similar cases

Fetch patient 15, compare their treatment with alternatives, and generate a chart of outcomes

Get the KPIs, identify the highest risk condition, and create a visualization of that metric
```

---

## 🧪 Edge Case & Error Handling Tests

```text
# Non-existent patient
Get patient 99999

# Invalid parameters
Generate a chart with invalid metric "xyz"

# Missing required fields
Add a patient record with only age 45

# Out of range values
Predict risk for a patient aged 150

# Empty results
Find patients with condition "NonExistentDisease"
```

---

## 💡 Tips for Testing

1. **Start Simple**: Test single-tool queries first to ensure each tool works
2. **Check RBAC**: Test with both `doctor` and `nurse` roles to verify access controls
3. **Multi-Tool Flow**: Use the ReAct agent (`Chatbot_react.py`) for multi-tool workflows
4. **Chart Verification**: After chart queries, verify the image renders correctly
5. **Error Messages**: Verify helpful error messages for invalid inputs
6. **SHAP Explanations**: Check that risk assessment includes feature importance explanations

