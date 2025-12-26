# MCP Inspector Configuration Guide

This guide explains how to connect MCP Inspector to your Hospital MCP Server running with Streamable HTTP transport.

---

## 📋 Prerequisites

1. **MCP Server Running**: Ensure your server is running on `http://localhost:8000/mcp`
   ```bash
   python server.py
   ```

2. **MCP Inspector Installed**: Install the MCP Inspector tool
   ```bash
   npx @anthropic/mcp-inspector
   # or install globally
   npm install -g @modelcontextprotocol/inspector
   ```

---

## 🚀 Quick Start

### Method 1: Using npx (Recommended)

```bash
# Start MCP Inspector and connect to your server
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

### Method 2: Using the Inspector UI

1. **Start the Inspector**
   ```bash
   npx @modelcontextprotocol/inspector
   ```

2. **Open in Browser**: Navigate to the URL shown (typically `http://localhost:5173`)

3. **Configure Connection**:
   - **Transport Type**: Select `Streamable HTTP`
   - **URL**: Enter `http://localhost:8000/mcp`
   - Click **Connect**

---

## ⚙️ Configuration Options

### Streamable HTTP Transport Settings

| Setting | Value |
|---------|-------|
| Transport | `streamable-http` |
| URL | `http://localhost:8000/mcp` |
| Method | `POST` |

### Connection Configuration JSON

If the inspector supports JSON configuration:

```json
{
  "transport": "streamable-http",
  "url": "http://localhost:8000/mcp"
}
```

---

## 🔧 Troubleshooting

### Issue: "Connection Failed" or "Cannot Connect"

**Cause**: MCP server not running or wrong port

**Solution**:
1. Verify server is running:
   ```bash
   # Check if server is listening
   curl http://localhost:8000/mcp
   ```
2. Check terminal for server startup message
3. Ensure no firewall blocking port 8000

### Issue: "Transport Not Supported"

**Cause**: Older MCP Inspector version

**Solution**:
```bash
# Update to latest version
npm update -g @modelcontextprotocol/inspector
# or
npx @modelcontextprotocol/inspector@latest
```

### Issue: "CORS Error" in Browser

**Cause**: Cross-origin requests blocked

**Solution**: The FastMCP server should handle CORS automatically. If issues persist:
1. Try accessing from `localhost` instead of `127.0.0.1`
2. Check server logs for CORS-related errors

### Issue: Tools Not Loading

**Cause**: Server initialization incomplete

**Solution**:
1. Wait a few seconds after server starts
2. Click "Refresh" in the Inspector
3. Check server logs for errors during tool registration

---

## 🧪 Testing Tools with Inspector

Once connected, you can test each tool directly:

### 1. List Available Tools

The Inspector should display all 15 available tools:
- `add_record`
- `update_record`
- `get_record`
- `risk_assessment`
- `compare_treatment`
- `similar_patients`
- `detect_anomaly`
- `get_record_summary`
- `kpi_overview`
- `length_of_stay_benchmark`
- `satisfaction_drivers_report`
- `predict_length_of_stay`
- `triage_queue`
- `counterfactual_readmission`
- `generate_chart`

### 2. Test Tool Execution

#### Test `get_record`
```json
{
  "patient_id": 5
}
```

#### Test `kpi_overview`
```json
{}
```
(No parameters required)

#### Test `risk_assessment`
```json
{
  "patient_attributes": {
    "Age": 65,
    "Gender": "Male",
    "Condition": "Diabetes",
    "Procedure": "Insulin Therapy",
    "Length_of_Stay": 5,
    "Cost": 8000
  }
}
```

#### Test `generate_chart`
```json
{
  "plot_type": "bar",
  "metric": "age",
  "group_by": "gender"
}
```

#### Test `counterfactual_readmission`
```json
{
  "patient_attributes": {
    "Age": 60,
    "Gender": "Female",
    "Condition": "Heart Disease",
    "Procedure": "Angioplasty",
    "Length_of_Stay": 7,
    "Cost": 25000
  },
  "changes_to_test": ["younger_age", "shorter_stay"]
}
```

### 3. View Tool Responses

- **Text Responses**: Displayed as formatted JSON
- **Image Responses**: Should render inline (for `generate_chart`)
- **Error Messages**: Shown with error details

---

## 📊 Inspector Features

### Tool Schema Inspection

View the full schema for each tool:
- Parameter names and types
- Required vs optional parameters
- Parameter descriptions
- Return type information

### Request/Response Logging

Monitor all communication:
- Request payloads
- Response data
- Timing information
- Error details

### Interactive Testing

- Modify parameters in real-time
- Re-run tool calls
- Compare responses

---

## 🔄 Alternative: Using curl for Testing

If MCP Inspector has issues, test directly with curl:

```bash
# Initialize session
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}'

# List tools
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}}'

# Call a tool
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 3, "params": {"name": "kpi_overview", "arguments": {}}}'
```

---

## 📝 Notes

1. **Session Management**: The Streamable HTTP transport creates a new session for each connection
2. **Tool Filtering**: Inspector shows all tools; RBAC filtering only applies in Chatbot clients
3. **Image Handling**: Chart images are returned as base64-encoded PNG data
4. **Concurrent Testing**: You can have Inspector and Streamlit client connected simultaneously

---

## 🔗 Related Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/docs)
- [MCP Inspector GitHub](https://github.com/modelcontextprotocol/inspector)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

