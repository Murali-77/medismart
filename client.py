import streamlit as st
import asyncio
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
from utils.intent_classifier import detect_intent_multilingual, welcome_message, unrelated_message
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE = SCRIPT_DIR / ".env"

print(f"Script directory: {SCRIPT_DIR}")
print(f"Looking for .env at: {ENV_FILE}")
print(f".env exists: {ENV_FILE.exists()}")

# Load environment variables BEFORE any caching
load_dotenv(ENV_FILE, override=True)

# Get API key and store it
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(f"GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")
if GROQ_API_KEY:
    print(f"GROQ_API_KEY value (first 10 chars): {GROQ_API_KEY[:10]}")
    print(f"GROQ_API_KEY length: {len(GROQ_API_KEY)}")
else:
    print("ERROR: GROQ_API_KEY is None!")

if not GROQ_API_KEY:
    st.error(f"GROQ_API_KEY not found in .env file at {ENV_FILE}!")
    st.stop()

# Set it in environment for any code that might look there
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Cache MCP client and agent setup
@st.cache_resource
def initialize_agent(_api_key: str):
    """
    Initialize the MCP agent and client.
    Note: _api_key is prefixed with _ to exclude it from Streamlit's hash calculation
    """
    print(f"[Inside initialize_agent] API Key first 10 chars: {_api_key[:10]}")
    
    # CRITICAL: Set the environment variable inside the cached function too
    # This ensures it's available when the LLM makes async calls
    os.environ["GROQ_API_KEY"] = _api_key
    
    config_file = "hospital_mcp.json"  
    client = MCPClient.from_config_file(config_file)
    
    # Create LLM - explicitly pass the API key AND set in environment
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=_api_key,
        groq_api_key=_api_key  # Some versions use this parameter
    )
    
    # Verify the LLM was created with the right key
    print(f"[LLM Created] Testing API key...")
    try:
        test_response = llm.invoke("Hi")
        print(f"[LLM Test] Success! Response: {test_response.content[:20]}")
    except Exception as e:
        print(f"[LLM Test] FAILED: {e}")
        raise
    
    # Create the agent with the LLM
    agent = MCPAgent(llm=llm, client=client, max_steps=10, memory_enabled=True)
    return agent, client

agent, client = initialize_agent(GROQ_API_KEY)

# Streamlit UI config
st.set_page_config(page_title="MCP Hospital Assistant", page_icon="🩺", layout="centered")
st.title("🧠 Hospital Data MCP Chatbot")

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

# Session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history in sidebar
for idx, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    st.sidebar.markdown(f"**{role.title()} {idx+1}:** {content[:60]}...")

# Chat input
user_input = st.chat_input("Ask something related to hospital data...")

if user_input:
    intent=detect_intent_multilingual(user_input)
    print(f"[Intent] {intent}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Handle user query - the system prompt is now included with the user input
        if intent=="hospital":
            try:
                response = asyncio.run(agent.run(user_input))
            except Exception as e:
                response = f"Error: {e}"
        elif intent=="greeting": response = welcome_message(user_input)
        else: response = unrelated_message(user_input)

        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})