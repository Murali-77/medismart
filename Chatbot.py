import streamlit as st
import asyncio
import time
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from mcp_use import MCPAgent, MCPClient
from utils.intent_classifier import detect_intent_multilingual, welcome_message, unrelated_message
from utils.plotting import get_latest_chart, clear_latest_chart
from pathlib import Path

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
        "triage_queue",
        # Visualization tool
        "generate_chart"
    ]
}

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
    # Collect all tools the user IS allowed to access
    allowed_tools = set()
    if user_role in ROLE_PERMISSIONS:
        allowed_tools.update(ROLE_PERMISSIONS[user_role])
    
    # Tools NOT in allowed list = disallowed
    disallowed = list(ALL_TOOLS - allowed_tools)
    
    return disallowed


# Load environment variables
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

# ============================================================================
# LLM CONFIGURATION - Easy to swap between providers
# ============================================================================
# Options: "azure_openai", "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai")

def get_llm():
    """
    Get the configured LLM based on LLM_PROVIDER environment variable.
    Makes it easy to swap between different LLM providers.
    """
    if LLM_PROVIDER == "azure_openai":
        
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
    
    elif LLM_PROVIDER == "groq":
        # GROQ Configuration
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in .env")
            st.stop()
        
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        return ChatGroq(
            model=groq_model,
            api_key=groq_api_key,
            groq_api_key=groq_api_key,
            temperature=0.7,
        )
    
    else:
        st.error(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'azure_openai' or 'groq'")
        st.stop()


# Cache resource AFTER authentication check
@st.cache_resource
def initialize_agent(_llm_provider: str, _disallowed_tools: tuple):
    """
    Initialize the MCP agent and client. 
    This expensive call is done only once per session & ONLY if user is logged in.
    """
    config_file = "hospital_mcp.json"
    client = MCPClient.from_config_file(config_file)

    # Get the configured LLM
    llm = get_llm()

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=10,
        memory_enabled=True,
        disallowed_tools=list(_disallowed_tools),
    )

    return agent, client


# 🚀 Initialize only for authorized users
user_role = st.session_state.get("roles")[0]
print(st.session_state)
disallowed_tools = get_disallowed_tools_for_user(user_role)
agent, client = initialize_agent(LLM_PROVIDER, tuple(disallowed_tools))


# -------------------------------
#          STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="MCP Hospital Assistant", page_icon="🩺", layout="centered")
st.title("🧠 Hospital Data MCP Chatbot")

# Show allowed tools based on role
allowed_tools_set = set()
if user_role in ROLE_PERMISSIONS:
    allowed_tools_set.update(ROLE_PERMISSIONS[user_role])

# Categorize tools for better sidebar display
TOOL_CATEGORIES = {
    "📋 Data Management": ["add_record", "update_record", "get_record", "get_record_summary"],
    "🔬 Analytics": ["kpi_overview", "length_of_stay_benchmark", "satisfaction_drivers_report", "detect_anomaly"],
    "🤖 ML Predictions": ["risk_assessment", "predict_length_of_stay", "counterfactual_readmission"],
    "🏥 Clinical Tools": ["compare_treatment", "similar_patients", "triage_queue"],
    "📊 Visualization": ["generate_chart"]
}

st.sidebar.markdown(f"**👤 Role:** `{user_role}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**🔓 Available Tools:**")

for category, tools in TOOL_CATEGORIES.items():
    category_tools = [t for t in tools if t in allowed_tools_set]
    if category_tools:
        st.sidebar.markdown(f"**{category}**")
        for tool in category_tools:
            st.sidebar.markdown(f"  - `{tool}`")

st.sidebar.divider()

# Sidebar controls
st.sidebar.header("🗂 Conversation History")

if st.sidebar.button("🧹 Clear Chat"):
    agent.clear_conversation_history()
    st.session_state.messages = []
    st.success("Conversation history cleared!")

if st.sidebar.button("🔒 Close MCP Sessions"):
    if client and client.sessions:
        asyncio.run(client.close_all_sessions())
        st.success("All MCP sessions closed.")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in sidebar
for idx, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    st.sidebar.markdown(f"**{role.title()} {idx+1}:** {content[:60]}...")

# Chat input
user_input = st.chat_input("Ask something related to hospital data...")

if user_input:
    intent = detect_intent_multilingual(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "intent": intent})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Record time before query to detect new charts
        query_start_time = time.time()

        if intent == "hospital":
            try:
                response = asyncio.run(agent.run(user_input))
            except Exception as e:
                response = f"Error: {e}"
        elif intent == "greeting":
            response = welcome_message(user_input)
        else:
            response = unrelated_message(user_input)

        # Display text response (remove attachment:// references as we'll show actual image)
        display_response = response
        if "attachment://" in response:
            # Clean up the markdown image reference since we'll show the actual image
            import re
            display_response = re.sub(r'!\[.*?\]\(attachment://.*?\)', '', response).strip()
            if not display_response:
                display_response = "Here is the generated chart:"
        
        message_placeholder.markdown(display_response)
        
        # Check if a chart was generated during this query
        chart_data = get_latest_chart()
        if chart_data:
            image_bytes, chart_title, chart_timestamp = chart_data
            # Only show chart if it was created after query started
            if chart_timestamp >= query_start_time - 1:  # 1 second tolerance
                chart_placeholder.image(image_bytes, caption=chart_title, use_container_width=True)
                clear_latest_chart()
                # Store chart info in message for history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": display_response,
                    "chart_title": chart_title
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": display_response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": display_response})
