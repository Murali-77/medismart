"""
Hospital MCP Chatbot with Multi-Tool Support

This version uses langchain-mcp-adapters and create_react_agent to support
multiple sequential/parallel tool calls per request. This enables workflows like:
- Get patient record → then run counterfactual analysis
- Query KPIs → then generate charts based on results

Uses Azure GPT-4o as the LLM.
"""
import streamlit as st
import asyncio
import time
import re
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.intent_classifier import detect_intent_multilingual, welcome_message, unrelated_message
from utils.plotting import get_latest_chart, clear_latest_chart

# Prevent agent initialization before auth
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# ⛔ If user is not logged in, block access immediately
if not st.session_state.get("authentication_status"):
    st.error("You must be logged in to access the Hospital MCP Chatbot.")
    st.stop()

# Only reach here if LOGGED IN ✔️

# ============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# ============================================================================
# Doctor: Full access to all tools including ML explanations and counterfactuals
# Nurse: Operational access - summaries, KPIs, benchmarks, triage (no risk model), charts

ROLE_PERMISSIONS = {
    "doctor": [
        # Existing tools
        "add_record",
        "update_record",
        "get_record",
        "risk_assessment",  # Now includes SHAP explanations
        "compare_treatment",
        "similar_patients",
        "detect_anomaly",
        # New analytics tools
        "get_record_summary",
        "kpi_overview",
        "length_of_stay_benchmark",
        "satisfaction_drivers_report",
        # New ML tools
        "predict_length_of_stay",
        "triage_queue",  # With include_risk_probability option
        "counterfactual_readmission",
        # New visualization tool
        "generate_chart"
    ],
    "nurse": [
        # Existing tools
        "add_record",
        "get_record",
        "similar_patients",
        "update_record",
        # New analytics tools (aggregated/operational)
        "get_record_summary",
        "kpi_overview",
        "length_of_stay_benchmark",
        # Triage (without risk model probability - enforced by not having risk_assessment)
        "triage_queue",
        # Visualization tool
        "generate_chart"
    ]
}

# All available MCP tools - MUST be updated when new tools are added
ALL_TOOLS = {
    # Existing tools
    "add_record",
    "update_record",
    "get_record",
    "risk_assessment",
    "compare_treatment",
    "similar_patients",
    "detect_anomaly",
    # New analytics tools
    "get_record_summary",
    "kpi_overview",
    "length_of_stay_benchmark",
    "satisfaction_drivers_report",
    # New ML tools
    "predict_length_of_stay",
    "triage_queue",
    "counterfactual_readmission",
    # New visualization tool
    "generate_chart"
}


def get_disallowed_tools_for_user(user_role: str) -> list[str]:
    """
    Calculate which tools should be DISALLOWED based on user roles.
    
    Args:
        user_role: Role assigned to the user (e.g., 'doctor', 'nurse')
    
    Returns:
        List of tool names that should be disallowed
    """
    allowed_tools = set()
    if user_role in ROLE_PERMISSIONS:
        allowed_tools.update(ROLE_PERMISSIONS[user_role])
    
    disallowed = list(ALL_TOOLS - allowed_tools)
    
    return disallowed


def get_allowed_tools_for_user(user_role: str) -> set[str]:
    """Get tools allowed for a specific role."""
    if user_role in ROLE_PERMISSIONS:
        return set(ROLE_PERMISSIONS[user_role])
    return set()


# Load environment variables
ENV_FILE = Path(__file__).parent.resolve() / ".env"
print(f"Loading .env from: {ENV_FILE}")
print(f".env file exists: {ENV_FILE.exists()}")
load_dotenv(ENV_FILE, override=True)

# ============================================================================
# LLM CONFIGURATION - Azure GPT-4o
# ============================================================================

def get_llm():
    """Get the Azure GPT-4o LLM instance."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0.3,  # Lower temperature for more consistent tool usage
    )


# ============================================================================
# MCP SERVER CONFIGURATION
# ============================================================================

MCP_SERVER_CONFIG = {
    "hospital": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http"
    }
}


# System prompt for the ReAct agent
SYSTEM_PROMPT = """You are a helpful hospital data assistant with access to various tools for managing patient records and generating insights.

IMPORTANT GUIDELINES:
1. You can call MULTIPLE tools in sequence to fulfill complex requests
2. When a user asks about a patient by ID, first use get_record to fetch their details
3. For counterfactual analysis, first get the patient record, then use that data for counterfactual_readmission
4. For risk assessment by patient ID, first get the record, then call risk_assessment
5. Always provide clear, concise explanations of the results
6. If a tool returns an error, explain it to the user and suggest alternatives

AVAILABLE WORKFLOWS:
- Patient lookup + analysis: get_record → risk_assessment / counterfactual_readmission / predict_length_of_stay
- Data exploration + visualization: kpi_overview → generate_chart
- Patient comparison: get_record (multiple) → compare_treatment / similar_patients

When generating charts, describe what the chart shows after it's generated."""


async def create_mcp_agent(allowed_tools: set[str]):
    """
    Create a ReAct agent with MCP tools using langchain-mcp-adapters.
    
    This agent supports multiple sequential tool calls, enabling complex workflows
    like: get_record → counterfactual_readmission
    
    Args:
        allowed_tools: Set of tool names the user is allowed to access
    
    Returns:
        tuple: (agent, client, tools) - The ReAct agent, MCP client, and filtered tools
    """
    # Create MCP client with server configuration
    # Note: A new session is created for each tool call automatically
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    
    # Get all tools from the MCP server (async call)
    all_tools = await client.get_tools()
    
    # Filter tools based on RBAC
    filtered_tools = [
        tool for tool in all_tools 
        if tool.name in allowed_tools
    ]
    
    # Get the LLM
    llm = get_llm()
    
    # Create the ReAct agent with filtered tools
    # ReAct agent handles multi-step reasoning and tool chaining automatically
    agent = create_react_agent(
        model=llm,
        tools=filtered_tools,
        prompt=SYSTEM_PROMPT
    )
    
    return agent, client, filtered_tools


async def run_agent_query(agent, query: str, message_history: list = None):
    """
    Run a query through the ReAct agent.
    
    The agent will automatically chain multiple tool calls if needed.
    
    Args:
        agent: The ReAct agent
        query: User's query
        message_history: Optional conversation history
    
    Returns:
        str: The agent's response
    """
    # Build messages list
    messages = []
    
    # Add conversation history if provided
    if message_history:
        for msg in message_history[-10:]:  # Keep last 10 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    # Add the current query
    messages.append(HumanMessage(content=query))
    
    # Invoke the agent
    result = await agent.ainvoke({"messages": messages})
    
    # Extract the final response
    if "messages" in result:
        # Get the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
    
    return "I apologize, but I couldn't generate a response. Please try again."


# ============================================================================
# STREAMLIT SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = None


async def initialize_or_get_agent(allowed_tools: set[str]):
    """Initialize the agent if not already done, or return existing one."""
    if not st.session_state.agent_initialized:
        agent, client, tools = await create_mcp_agent(allowed_tools)
        st.session_state.react_agent = agent
        st.session_state.mcp_client = client
        st.session_state.available_tools = [t.name for t in tools]
        st.session_state.agent_initialized = True
        return agent, client, tools
    return st.session_state.react_agent, st.session_state.mcp_client, None


def cleanup_mcp_client():
    """Clean up the MCP client connection."""
    # Note: MultiServerMCPClient creates a new session for each tool call,
    # so we just need to reset the state
    st.session_state.mcp_client = None
    st.session_state.agent_initialized = False


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    # Get user role
    user_role = st.session_state.get("roles", ["doctor"])[0]
    allowed_tools = get_allowed_tools_for_user(user_role)
    
    # Page config
    st.set_page_config(
        page_title="MCP Hospital Assistant (ReAct)", 
        page_icon="🩺", 
        layout="centered"
    )
    st.title("🧠 Hospital Data MCP Chatbot")
    st.caption("**ReAct Agent** - Supports multi-tool workflows")
    
    # Sidebar - Role and Tools
    st.sidebar.markdown(f"**👤 Role:** `{user_role}`")
    st.sidebar.markdown("**🔄 Agent Type:** ReAct (Multi-Tool)")
    st.sidebar.markdown("---")
    
    # Categorize tools for better sidebar display
    TOOL_CATEGORIES = {
        "📋 Data Management": ["add_record", "update_record", "get_record", "get_record_summary"],
        "🔬 Analytics": ["kpi_overview", "length_of_stay_benchmark", "satisfaction_drivers_report", "detect_anomaly"],
        "🤖 ML Predictions": ["risk_assessment", "predict_length_of_stay", "counterfactual_readmission"],
        "🏥 Clinical Tools": ["compare_treatment", "similar_patients", "triage_queue"],
        "📊 Visualization": ["generate_chart"]
    }
    
    st.sidebar.markdown("**🔓 Available Tools:**")
    for category, tools in TOOL_CATEGORIES.items():
        category_tools = [t for t in tools if t in allowed_tools]
        if category_tools:
            st.sidebar.markdown(f"**{category}**")
            for tool in category_tools:
                st.sidebar.markdown(f"  - `{tool}`")
    
    st.sidebar.divider()
    
    # Sidebar controls
    st.sidebar.header("🗂 Session Controls")
    
    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.success("Conversation history cleared!")
        st.rerun()
    
    if st.sidebar.button("🔄 Reconnect MCP"):
        cleanup_mcp_client()
        st.success("MCP connection will be re-established on next query.")
        st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display chart if stored
            if msg.get("has_chart") and msg.get("chart_bytes"):
                st.image(msg["chart_bytes"], caption=msg.get("chart_title", "Chart"))
    
    # Chat input
    user_input = st.chat_input("Ask something related to hospital data...")
    
    if user_input:
        # Detect intent
        intent = detect_intent_multilingual(user_input)
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input, 
            "intent": intent
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process with assistant
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Record time before query to detect new charts
            query_start_time = time.time()
            
            if intent == "hospital":
                try:
                    with st.spinner("🔄 Processing with ReAct agent..."):
                        # Initialize agent if needed and run query
                        async def process_query():
                            agent, client, _ = await initialize_or_get_agent(allowed_tools)
                            response = await run_agent_query(
                                agent, 
                                user_input, 
                                st.session_state.messages[:-1]  # Exclude current message
                            )
                            return response
                        
                        response = asyncio.run(process_query())
                        
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(f"Agent error: {e}")
            
            elif intent == "greeting":
                response = welcome_message(user_input)
            else:
                response = unrelated_message(user_input)
            
            # Display text response (remove attachment:// references)
            display_response = response
            if "attachment://" in response:
                display_response = re.sub(r'!\[.*?\]\(attachment://.*?\)', '', response).strip()
                if not display_response:
                    display_response = "Here is the generated chart:"
            
            message_placeholder.markdown(display_response)
            
            # Check if a chart was generated during this query
            chart_data = get_latest_chart()
            chart_bytes = None
            chart_title = None
            
            if chart_data:
                image_bytes, title, chart_timestamp = chart_data
                # Only show chart if it was created after query started
                if chart_timestamp >= query_start_time - 1:  # 1 second tolerance
                    chart_placeholder.image(image_bytes, caption=title, use_container_width=True)
                    chart_bytes = image_bytes
                    chart_title = title
                    clear_latest_chart()
            
            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_response,
                "has_chart": chart_bytes is not None,
                "chart_bytes": chart_bytes,
                "chart_title": chart_title
            })


if __name__ == "__main__":
    main()

