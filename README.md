# MediSmart - Hospital Assistant MCP Server

A comprehensive Model Context Protocol (MCP) server for hospital patient data management, featuring advanced analytics, ML-powered predictions with SHAP explanations, data visualization, and custom Streamlit-based chatbot clients with multilingual intent classification and RBAC.

## 🌟 Features

### 📊 **Data Management**
- **Patient Record Management**: Add, update, and retrieve patient records
- **PostgreSQL-based Storage**: Efficient data storage with SQLAlchemy ORM
- **Data Validation**: Robust input validation and error handling
- **Record Summarization**: Quick patient summaries and overviews

### 📈 **Machine Learning & Analytics**
- **Risk Assessment with SHAP**: Predict patient readmission risk with explainable AI
- **Length of Stay Prediction**: ML-powered hospital stay duration estimation
- **Treatment Comparison**: Compare effectiveness of different medical procedures
- **Similar Patient Search**: Find historical cases similar to current patients
- **Counterfactual Analysis**: What-if scenarios for readmission prevention
- **Anomaly Detection**: Identify data quality issues and outliers
- **KPI Dashboard**: Key performance indicators and hospital metrics
- **Satisfaction Analysis**: Patient satisfaction drivers report

### 📊 **Data Visualization**
- **Chart Generation**: Bar, line, pie, histogram, and boxplot charts
- **Metric Analysis**: Visualize billing, admissions, age, length of stay
- **Grouped Analysis**: Group data by condition, procedure, gender, outcome

### 🤖 **Custom Client Interfaces**
- **Streamlit Chatbot** (`Chatbot.py`): Single-tool interactive chat interface
- **ReAct Agent Chatbot** (`Chatbot_react.py`): Multi-tool workflow support
- **Intent Routing**: Automatically classifies queries as hospital-related, greetings, or unrelated
- **Conversation Memory**: Maintains chat history with clear and close options
- **RBAC**: Login/Signup functionality with role-based conditional tool access

### 🔧 **MCP Tools Available**

| Tool | Category | Description |
|------|----------|-------------|
| `add_record` | Data Management | Add new patient records to the system |
| `update_record` | Data Management | Update existing patient information |
| `get_record` | Data Management | Query patient records with flexible criteria |
| `get_record_summary` | Data Management | Get a quick summary of patient records |
| `risk_assessment` | ML Predictions | Assess readmission risk with SHAP explanations |
| `predict_length_of_stay` | ML Predictions | Predict hospital stay duration |
| `counterfactual_readmission` | ML Predictions | What-if analysis for readmission prevention |
| `compare_treatment` | Clinical Tools | Compare treatment efficacy for conditions |
| `similar_patients` | Clinical Tools | Find similar historical patient cases |
| `triage_queue` | Clinical Tools | Patient prioritization queue |
| `detect_anomaly` | Analytics | Identify anomalies and data quality issues |
| `kpi_overview` | Analytics | Hospital key performance indicators |
| `length_of_stay_benchmark` | Analytics | LOS benchmarking by condition/procedure |
| `satisfaction_drivers_report` | Analytics | Patient satisfaction analysis |
| `generate_chart` | Visualization | Generate various data visualization charts |

### 🔐 **Role-Based Access Control (RBAC)**

| Role | Allowed Tools |
|------|---------------|
| **Doctor** | All 15 tools including ML predictions, counterfactual analysis, and SHAP explanations |
| **Nurse** | 11 tools (excludes risk_assessment, satisfaction_drivers_report, predict_length_of_stay, counterfactual_readmission) |

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or uv package manager
- PostgreSQL database (or modify for other databases)
- Azure OpenAI API access (for GPT-4o) or GROQ API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JashwanthSA/medismart
   cd file_reader
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Train Required Models**
   
   Before using the system, train both ML models:
   
   ```bash
   # Train the intent classifier (for chatbot routing)
   python utils/train_intent_model.py
   
   # Train the risk assessment model
   python utils/train_risk_model.py
   
   # Train the length of stay model
   python utils/train_los_model.py
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # LLM Provider: "azure_openai" or "groq"
   LLM_PROVIDER=azure_openai
   
   # Azure OpenAI Configuration (if using Azure)
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_API_VERSION=2024-08-01-preview
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   
   # GROQ Configuration (if using GROQ)
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.3-70b-versatile
   
   # Database Configuration
   DB_HOST=localhost
   DB_NAME=hospital_db
   DB_USER=your_user
   DB_PASSWORD=your_password
   ```

5. **Run the MCP server**
   
   The server supports two transport modes:
   
   **A. Streamable HTTP Transport** (for custom clients):
   ```bash
   python server.py
   ```
   This starts the server at `http://localhost:8000/mcp`
   
   **B. Stdio Transport** (for Claude Desktop integration):
   
   Modify `server.py` to use stdio transport:
   ```python
   # In server.py, change the last line from:
   mcp.run(transport="streamable-http")
   # To:
   mcp.run(transport="stdio")
   ```

6. **Launch the Custom Client**
   
   In a separate terminal, run one of the Streamlit chatbots:
   
   ```bash
   # Standard chatbot (single-tool per request)
   streamlit run Chatbot.py
   
   # ReAct agent chatbot (multi-tool workflows)
   streamlit run Chatbot_react.py --server.port 8503
   ```

## 🖥️ Client Options

### Option 1: Standard Chatbot (`Chatbot.py`)

Single-tool execution per request. Best for simple queries.

**Features:**
- 🌍 Multilingual Support
- 🎯 Smart Intent Routing
- 💬 Conversation Memory
- 🔐 RBAC Integration
- 📊 Chart Rendering
- 🔄 Swappable LLM (Azure OpenAI / GROQ)

### Option 2: ReAct Agent Chatbot (`Chatbot_react.py`)

Multi-tool workflow support using LangChain's ReAct agent.

**Features:**
- 🔄 **Multi-Tool Chaining**: Execute multiple tools in sequence
- 🧠 **Intelligent Routing**: Agent decides which tools to call
- 📋 **Complex Workflows**: e.g., "Get patient 5 and run counterfactual analysis"

**Example Multi-Tool Queries:**
```text
Get patient 5's record and assess their readmission risk
Fetch patient 10 and run a counterfactual analysis
Show KPIs and then generate a bar chart for admissions by condition
```

### Option 3: Claude Desktop (Stdio Transport)

Direct integration with Claude Desktop for native MCP tool access.

## 📋 Data Schema

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

### Risk Assessment with SHAP Explanations
```python
patient_attributes = {
    "Age": 65,
    "Gender": "Male",
    "Condition": "Diabetes",
    "Procedure": "Insulin Therapy",
    "Length_of_Stay": 5,
    "Cost": 8000
}
risk_result = risk_assessment(patient_attributes)
# Returns prediction with SHAP feature importance explanations
```

### Generate Visualization
```python
chart_result = generate_chart(
    plot_type="bar",
    metric="age",
    group_by="condition"
)
# Returns PNG image of the chart
```

### Multi-Tool Workflow (ReAct Agent)
```text
User: "Get patient 5's record and run counterfactual analysis"

Agent executes:
1. get_record(patient_id=5) → Gets patient details
2. counterfactual_readmission(patient_attributes=...) → Analyzes scenarios
3. Returns combined insights
```

## 🏗️ Architecture

```
file_reader/
├── Chatbot.py                   # Standard Streamlit chatbot (single-tool)
├── Chatbot_react.py             # ReAct agent chatbot (multi-tool)
├── server.py                    # MCP server implementation
├── hospital_mcp.json            # MCP client configuration
├── .env                         # Environment variables
├── data/                        # Data storage
│   ├── hospital-data-analysis.csv
│   ├── patient_risk_model.joblib    # Risk assessment model
│   ├── patient_los_model.joblib     # Length of stay model
│   ├── intent_clf_transformer.joblib
│   ├── embed_model/
│   └── charts/                      # Generated chart images
├── utils/                       # Core functionality modules
│   ├── file_reader.py           # Data I/O operations
│   ├── risk.py                  # Risk assessment with SHAP
│   ├── los.py                   # Length of stay prediction
│   ├── analytics.py             # KPIs, benchmarks, summaries
│   ├── triage.py                # Patient triage queue
│   ├── counterfactual.py        # What-if analysis
│   ├── plotting.py              # Chart generation
│   ├── compare_efficacy.py      # Treatment comparison
│   ├── similar.py               # Similar patient search
│   ├── anamoly.py               # Anomaly detection
│   ├── train_risk_model.py      # Risk model training
│   ├── train_los_model.py       # LOS model training
│   ├── train_intent_model.py    # Intent classifier training
│   └── intent_classifier.py     # Intent detection
└── pyproject.toml
```

## 🤖 Machine Learning Models

### Risk Assessment Model
- **Algorithm**: Random Forest Classifier
- **Features**: Age, Gender, Condition, Procedure, Length of Stay, Cost
- **Target**: Readmission prediction (Yes/No)
- **Explainability**: SHAP values for feature importance
- **Training**: `python utils/train_risk_model.py`

### Length of Stay Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Age, Gender, Condition, Procedure, Cost
- **Target**: Length of Stay (days)
- **Explainability**: SHAP values for feature importance
- **Training**: `python utils/train_los_model.py`

### Intent Classification Model
- **Algorithm**: Logistic Regression with Sentence Transformers
- **Embeddings**: Multilingual paraphrase model
- **Categories**: `hospital`, `greeting`, `unrelated`
- **Training**: `python utils/train_intent_model.py`

## 🧪 Testing

### Using MCP Inspector

Connect MCP Inspector to test tools directly:

```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

See `docs/mcp_inspector_guide.md` for detailed setup instructions.

### Test Queries

See `docs/testing_queries.md` for comprehensive test queries including:
- Single-tool test cases
- Multi-tool workflow examples
- Edge cases and error handling

## 🔧 Configuration

### LLM Provider Selection

Set `LLM_PROVIDER` in `.env`:
- `azure_openai`: Uses Azure GPT-4o (recommended for multi-tool workflows)
- `groq`: Uses GROQ Llama 3.3 70B (faster, but smaller context)

### MCP Server Configuration

**For Custom Client (HTTP Transport)** - `hospital_mcp.json`:
```json
{
  "mcpServers": {
    "hospital": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable_http"
    }
  }
}
```

**For Claude Desktop (Stdio Transport)**:
```json
{
  "mcpServers": {
    "hospital-patient-data": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

## 📊 Sample Data

Sample dataset: [Hospital Dataset on Kaggle](https://www.kaggle.com/datasets/blueblushed/hospital-dataset-for-practice/data)

- **989 patient records**
- **10 Medical Conditions**: Heart Disease, Diabetes, Cancer, Stroke, etc.
- **15+ Procedures**: Angioplasty, Insulin Therapy, Surgery, etc.
- **Diverse Demographics**: Age range 25-78, both genders

## 🛠️ Development

### Adding New Tools

1. Create function in appropriate `utils/` module
2. Add `@mcp.tool()` decorator in `server.py`
3. Update RBAC permissions in `Chatbot.py` and `Chatbot_react.py`
4. Add test queries to `docs/testing_queries.md`

### Retraining Models

```bash
# After updating training data
python utils/train_risk_model.py
python utils/train_los_model.py
python utils/train_intent_model.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.
