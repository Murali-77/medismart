# MediSmart - Hospital Assistant MCP Server

A comprehensive Model Context Protocol (MCP) server for hospital patient data management, featuring advanced analytics, risk assessment, and treatment efficacy comparison capabilities.

## 🌟 Features

### 📊 **Data Management**
- **Patient Record Management**: Add, update, and retrieve patient records
- **MySQL-based Storage**: Efficient data storage with pandas integration
- **Data Validation**: Robust input validation and error handling

### 📈 **Machine Learning & Analytics**
- **Risk Assessment**: Predict patient readmission risk using Random Forest models
- **Treatment Comparison**: Compare effectiveness of different medical procedures
- **Similar Patient Search**: Find historical cases similar to current patients
- **Anomaly Detection**: Identify data quality issues and outliers

### 🔧 **MCP Tools Available**

| Tool | Function | Description |
|------|----------|-------------|
| `add_record` | Data Ingestion | Add new patient records to the system |
| `update_record` | Data Transformation | Update existing patient information |
| `get_record` | Data Retrieval | Query patient records with flexible criteria |
| `risk_assessment` | Predictive Analytics | Assess readmission risk for patients |
| `compare_treatment` | Comparative Analysis | Compare treatment efficacy for conditions |
| `similar_patients` | Pattern Matching | Find similar historical patient cases |
| `detect_anomaly` | Data Quality | Identify anomalies and data quality issues |

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd file_reader
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run the MCP server**
   ```bash
   python main.py
   ```

## 📋 Data Schema

The system works with hospital patient data containing the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `Patient_ID` | Integer | Unique patient identifier |
| `Age` | Integer | Patient age |
| `Gender` | String | Patient gender (Male/Female) |
| `Condition` | String | Medical condition |
| `Procedure` | String | Medical procedure performed |
| `Cost` | Float | Treatment cost in USD |
| `Length_of_Stay` | Integer | Hospital stay duration (days) |
| `Readmission` | String | Readmission status (Yes/No) |
| `Outcome` | String | Treatment outcome (Recovered/Stable) |
| `Satisfaction` | Integer | Patient satisfaction score (1-5) |

## 🔍 Usage Examples

### Adding a Patient Record
```python
patient_data = {
    "Age": 45,
    "Gender": "Female",
    "Condition": "Heart Disease",
    "Procedure": "Angioplasty",
    "Cost": 15000,
    "Length_of_Stay": 5,
    "Readmission": "No",
    "Outcome": "Recovered",
    "Satisfaction": 4
}
result = add_record(patient_data)
```

### Risk Assessment
```python
patient_attributes = {
    "Age": 65,
    "Gender": "Male",
    "Condition": "Diabetes",
    "Procedure": "Insulin Therapy",
    "Length_of_Stay": 3
}
risk_result = risk_assessment(patient_attributes)
# Returns: {"predicted_readmission": "No", "probability_of_readmission": "0.23"}
```

### Treatment Comparison
```python
condition = "Heart Disease"
procedures = ["Angioplasty", "Cardiac Catheterization"]
comparison = compare_treatment(condition, procedures)
```

## 🏗️ Architecture

```
file_reader/
├── server.py # MCP server implementation
├── data/ # Data storage
│ ├── hospital-data-analysis.csv
│ └── patient_risk_model.joblib
├── utils/ # Core functionality modules
│ ├── file_reader.py # Data I/O operations
│ ├── risk.py # Risk assessment & ML
│ ├── compare_efficacy.py # Treatment comparison
│ ├── similar.py # Similar patient search
│ ├── anamoly.py # Anomaly detection
│ └── train_model.py # Model training utilities
└── pyproject.toml # Project configuration
```


## 🤖 Machine Learning Features

### Risk Assessment Model
- **Algorithm**: Random Forest Classifier
- **Features**: Age, Gender, Condition, Procedure, Length of Stay
- **Target**: Readmission prediction
- **Accuracy**: Trained on 989 patient records
- **Auto-training**: Automatically trains if model file is missing

### Data Analytics
- **Treatment Efficacy**: Success rates, costs, and outcomes comparison
- **Pattern Recognition**: Find similar patient cases using key attributes
- **Anomaly Detection**: Z-score based outlier detection for numerical data
- **Quality Assurance**: Comprehensive data validation and error handling

## 🤝 Configuration

### MCP Server Settings
The server runs on the default MCP configuration. You can customize:
- Server name: "Hospital Patient Data Management System"
- Tool descriptions and parameters
- Data file paths and formats

### Model Configuration
- **Model File**: `data/patient_risk_model.joblib`
- **Training Data**: `data/hospital-data-analysis.csv`
- **Features**: Configurable feature set for risk assessment
- **Thresholds**: Adjustable parameters for anomaly detection

### Sample client json configuration
```
{
  "mcpServers": {
    "hospital-patient-data": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "server.py"
      ],
      "cwd": "The current directory of the code files",
      "env": {}
    }
  }
}
```

## 📊 Sample Data
Sample dataset link: [Click here](https://www.kaggle.com/datasets/blueblushed/hospital-dataset-for-practice/data)

This comprehensive dataset contains 989 patient records covering:
- **10 Medical Conditions**: Heart Disease, Diabetes, Cancer, Stroke, etc.
- **15+ Procedures**: Angioplasty, Insulin Therapy, Surgery, etc.
- **Diverse Demographics**: Age range 25-78, both genders
- **Realistic Metrics**: Costs, stay durations, outcomes, satisfaction scores

## 🛠️ Development

### Adding New Tools
1. Create a new function in the appropriate `utils/` module
2. Add the `@mcp.tool()` decorator in `server.py`
3. Update documentation and examples

### Extending the Model
1. Modify `utils/train_model.py` for new ML algorithms
2. Update feature engineering in `utils/risk.py`
3. Retrain and save the model using `utils/train_model.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request