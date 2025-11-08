# MediSmart - Hospital Assistant MCP Server

A comprehensive Model Context Protocol (MCP) server for hospital patient data management, featuring advanced analytics, risk assessment, treatment efficacy comparison capabilities, and a custom Streamlit-based chatbot client with multilingual intent classification.

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
- **Intent Classification**: Multilingual intent detection to route queries appropriately

### 🤖 **Custom Client Interface**
- **Streamlit Chatbot**: Interactive web-based chat interface (`client.py`)
- **Multilingual Support**: Detects user language and responds accordingly
- **Intent Routing**: Automatically classifies queries as hospital-related, greetings, or unrelated
- **Conversation Memory**: Maintains chat history with clear and close options

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
- GROQ API Key (for the custom client)

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

3. **Train the Intent Classifier Model** (Required for custom client)
   
   Before using the custom Streamlit client, you must train the intent classification model:
   
   ```bash
   cd utils
   python train_intent_model.py
   cd ..
   ```
   
   This will:
   - Load training data from `data/intent_data.jsonl`
   - Train a multilingual intent classifier using sentence transformers
   - Save the trained model to `data/intent_clf_transformer.joblib`
   - Save the embedding model to `data/embed_model/`

4. **Set up environment variables** (For custom client)
   
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the MCP server**
   
   The server supports two transport modes:
   
   **A. Streamable HTTP Transport** (for custom clients like `client.py`):
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

6. **Launch the Custom Client** (Optional)
   
   In a separate terminal, run the Streamlit chatbot:
   ```bash
   streamlit run client.py
   ```
   
   This will open a web browser with the interactive chat interface at `http://localhost:8501`

## 🖥️ Usage Modes

### Mode 1: Custom Streamlit Client (HTTP Transport)

The custom client provides an interactive web-based chat interface with the following features:

**Features:**
- 🌍 **Multilingual Support**: Automatically detects and responds in 20+ languages
- 🎯 **Smart Intent Routing**: Classifies queries as hospital-related, greetings, or unrelated
- 💬 **Conversation Memory**: Maintains chat history throughout the session
- 🧠 **Powered by GROQ**: Fast LLM inference using Llama 3.3 70B model
- 🔧 **MCP Integration**: Seamlessly calls MCP tools for hospital data operations

**Setup Requirements:**
1. Train the intent classifier: `python utils/train_intent_model.py`
2. Create `.env` file with your GROQ API key
3. Start MCP server in HTTP mode: `python server.py`
4. Launch client: `streamlit run client.py`

**User Experience:**
- Type queries in any supported language
- System automatically routes hospital queries to MCP tools
- Greetings receive friendly responses in the detected language
- Non-hospital queries are politely declined
- View conversation history in the sidebar
- Clear chat or close MCP sessions anytime

### Mode 2: Claude Desktop Integration (Stdio Transport)

For using the MCP server directly with Claude Desktop:

**Setup Requirements:**
1. Modify `server.py` to use stdio transport:
   ```python
   mcp.run(transport="stdio")  # Change from "streamable-http"
   ```
2. Add configuration to Claude Desktop's MCP settings file
3. Restart Claude Desktop

**User Experience:**
- Interact through Claude Desktop's native interface
- Claude AI directly invokes MCP tools as needed
- No additional client setup required
- Full access to all MCP tools

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
├── client.py                    # Streamlit-based custom chatbot client
├── server.py                    # MCP server implementation (HTTP/stdio)
├── hospital_mcp.json            # MCP client configuration for HTTP transport
├── .env                         # Environment variables (GROQ_API_KEY)
├── data/                        # Data storage
│   ├── hospital-data-analysis.csv
│   ├── patient_risk_model.joblib
│   ├── intent_data.jsonl        # Training data for intent classifier
│   ├── intent_clf_transformer.joblib  # Trained intent model
│   └── embed_model/             # Sentence transformer embeddings
├── utils/                       # Core functionality modules
│   ├── file_reader.py           # Data I/O operations
│   ├── risk.py                  # Risk assessment & ML
│   ├── compare_efficacy.py      # Treatment comparison
│   ├── similar.py               # Similar patient search
│   ├── anamoly.py               # Anomaly detection
│   ├── train_model.py           # Patient risk model training
│   ├── train_intent_model.py    # Intent classifier training
│   └── intent_classifier.py     # Intent detection & multilingual support
└── pyproject.toml               # Project configuration
```


## 🤖 Machine Learning Features

### Risk Assessment Model
- **Algorithm**: Random Forest Classifier
- **Features**: Age, Gender, Condition, Procedure, Length of Stay
- **Target**: Readmission prediction
- **Accuracy**: Trained on 989 patient records
- **Auto-training**: Automatically trains if model file is missing

### Intent Classification Model
- **Algorithm**: Logistic Regression with Sentence Transformers
- **Embeddings**: Multilingual paraphrase model (paraphrase-multilingual-MiniLM-L12-v2)
- **Supported Languages**: English, Spanish, French, German, Italian, Hindi, Tamil, Chinese, Japanese, Korean, and more
- **Intent Categories**: 
  - `hospital`: Hospital/medical-related queries
  - `greeting`: Welcome and conversational greetings
  - `unrelated`: Non-hospital queries
- **Training Required**: Must run `utils/train_intent_model.py` before using the custom client

### Data Analytics
- **Treatment Efficacy**: Success rates, costs, and outcomes comparison
- **Pattern Recognition**: Find similar patient cases using key attributes
- **Anomaly Detection**: Z-score based outlier detection for numerical data
- **Quality Assurance**: Comprehensive data validation and error handling

## 🤝 Configuration

### MCP Server Settings
The server supports two transport modes depending on your use case:

1. **Streamable HTTP Transport** (Default in `server.py`):
   - Used for custom client integration (e.g., `client.py`)
   - Server URL: `http://localhost:8000/mcp`
   - Configuration file: `hospital_mcp.json`

2. **Stdio Transport**:
   - Used for Claude Desktop integration
   - Requires modifying `server.py` to use `transport="stdio"`
   - Configure in Claude Desktop's MCP settings

### Model Configuration
- **Risk Model File**: `data/patient_risk_model.joblib`
- **Intent Model File**: `data/intent_clf_transformer.joblib`
- **Embedding Model**: `data/embed_model/`
- **Training Data**: `data/hospital-data-analysis.csv` and `data/intent_data.jsonl`
- **Features**: Configurable feature set for risk assessment
- **Thresholds**: Adjustable parameters for anomaly detection

### Client Configuration
The custom Streamlit client (`client.py`) requires:
- `.env` file with `GROQ_API_KEY`
- `hospital_mcp.json` configuration file
- Trained intent classifier model (run `utils/train_intent_model.py`)
- MCP server running on HTTP transport

### Sample MCP Configuration Files

#### For Custom Client (HTTP Transport) - `hospital_mcp.json`:
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

#### For Claude Desktop (Stdio Transport):
```json
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

**Note**: When using with Claude Desktop, ensure `server.py` is configured with `transport="stdio"` instead of `transport="streamable-http"`.

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

### Extending the Models
1. **Risk Assessment Model**: 
   - Modify `utils/train_model.py` for new ML algorithms
   - Update feature engineering in `utils/risk.py`
   - Retrain and save the model using `utils/train_model.py`

2. **Intent Classifier Model**:
   - Add new training examples to `data/intent_data.jsonl`
   - Modify intent categories in `utils/train_intent_model.py`
   - Retrain the model: `python utils/train_intent_model.py`
   - Update intent handling logic in `client.py`

### Working with the Custom Client
The Streamlit client (`client.py`) features:
- **Session Management**: MCP sessions with conversation history
- **Async Processing**: Handles long-running MCP tool calls
- **Intent Detection**: Routes queries based on detected intent
- **Multilingual Responses**: Automatic language detection and translation
- **Error Handling**: Graceful handling of API and connection errors

To customize the client:
- Modify UI components in `client.py`
- Update intent routing logic
- Add new response handlers
- Configure LLM parameters (model, temperature, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request