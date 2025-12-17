import streamlit as st
import asyncio
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
from utils.intent_classifier import detect_intent_multilingual, welcome_message, unrelated_message
from pathlib import Path

# Prevent agent initialization before auth
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# ⛔ If user is not logged in, block access immediately
if not st.session_state.get("authentication_status"):
    st.error("You must be logged in to access the Hospital MCP Chatbot.")
    st.stop()

# Only reach here if LOGGED IN ✔️

ROLE_PERMISSIONS = {
    "doctor": [
        "add_record",
        "update_record",
        "get_record",
        "risk_assessment",
        "compare_treatment",
        "similar_patients",
        "detect_anomaly"
    ],
    "nurse": [
        "add_record",
        "get_record",
        "similar_patients",
        "update_record"
    ]
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
    
    # All possible tools (update this list based on your actual MCP tools)
    all_tools = set([
        "add_record",
        "update_record",
        "get_record",
        "risk_assessment",
        "compare_treatment",
        "similar_patients",
        "detect_anomaly"
    ])
    
    # Tools NOT in allowed list = disallowed
    disallowed = list(all_tools - allowed_tools)
    
    return disallowed

# Load environment variables
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / ".env"
load_dotenv(ENV_FILE, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(f"GROQ_API_KEY not found in .env at {ENV_FILE}")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY



# Cache resource AFTER authentication check
@st.cache_resource
def initialize_agent(_api_key: str, _disallowed_tools: tuple):
    """
    Initialize the MCP agent and client. 
    This expensive call is done only once per session & ONLY if user is logged in.
    """
    os.environ["GROQ_API_KEY"] = _api_key

    config_file = "hospital_mcp.json"
    client = MCPClient.from_config_file(config_file)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=_api_key,
        groq_api_key=_api_key
    )

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=10,
        memory_enabled=True,
        disallowed_tools=list(_disallowed_tools),
        # system_prompt_template=SYSTEM_PROMPT_TEMPLATE
    )

    return agent, client


# 🚀 Initialize only for authorized users
user_role = st.session_state.get("roles")[0]
print(st.session_state)
disallowed_tools = get_disallowed_tools_for_user(user_role)
agent, client = initialize_agent(GROQ_API_KEY, tuple(disallowed_tools))


# -------------------------------
#          STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="MCP Hospital Assistant", page_icon="🩺", layout="centered")
st.title("🧠 Hospital Data MCP Chatbot")

# Show allowed tools based on role
allowed_tools_set = set()
if user_role in ROLE_PERMISSIONS:
    allowed_tools_set.update(ROLE_PERMISSIONS[user_role])

st.sidebar.markdown("**🔓 Available Tools:**")
if allowed_tools_set:
    for tool in sorted(allowed_tools_set):
        st.sidebar.markdown(f"- `{tool}`")
else:
    st.sidebar.markdown("_No tools available_")

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
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if intent == "hospital":
            try:
                response = asyncio.run(agent.run(user_input))
            except Exception as e:
                response = f"Error: {e}"
        elif intent == "greeting":
            response = welcome_message(user_input)
        else:
            response = unrelated_message(user_input)

        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})