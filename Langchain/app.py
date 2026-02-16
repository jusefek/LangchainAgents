import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import os

# Page Config
st.set_page_config(page_title="Gemini Agent", page_icon="🤖")

# --- Configuration ---
with st.sidebar:
    st.title("🤖 Configuration")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
        
    st.markdown("---")
    st.subheader("Model Settings")
    
    # Dynamic Model Discovery
    available_models = []
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    # Clean up model name (remove 'models/' prefix if present for display, but keep for usage if needed)
                    # actually LangChain usually expects just the name or 'models/name' depending on version.
                    # 'models/gemini-1.5-flash' is standard for the SDK.
                    available_models.append(m.name)
        except Exception as e:
            st.error(f"Error fetching models: {e}")
            
    if available_models:
        # Try to find a good default
        default_index = 0
        for i, m in enumerate(available_models):
            if "gemini-1.5" in m:
                default_index = i
                break
        
        selected_model = st.selectbox("Select Available Model", available_models, index=default_index)
        model_name = selected_model
    else:
        # Fallback if discovery fails or no key yet
        model_name = st.text_input("Model Name (e.g. gemini-1.5-flash)", value="gemini-1.5-flash")
        if api_key:
             st.warning("Could not automatically list models. Please type one manually.")

# --- Tools ---
@tool
def calculate_word_length(word: str) -> int:
    """Calculates the length of a given word."""
    return len(word)

@tool
def power_calculation(base: float, exponent: float) -> float:
    """Calculates the power of a base raised to an exponent."""
    return base ** exponent

tools = [calculate_word_length, power_calculation]
tools_map = {t.name: t for t in tools}

# --- Main App ---
st.title("🤖 Google Gemini 'Lite' Agent")

with st.expander("ℹ️ **Capabilities**", expanded= False):
    st.info("""
    This simplified agent uses direct tool calling.
    - **Tools**: Word Length, Power Calculation.
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        role = "user"
        content = msg.content
    elif isinstance(msg, AIMessage):
        role = "assistant"
        content = msg.content or ""
        if not content and msg.tool_calls:
            content = f"🛠️ *Calling tools: {', '.join([tc['name'] for tc in msg.tool_calls])}* ..."
    elif isinstance(msg, ToolMessage):
        role = "tool"
        content = f"✅ *Tool Result:* {msg.content}"
    else:
        role = "assistant"
        content = str(msg)
    
    if role != "tool": 
        with st.chat_message(role):
            st.markdown(content)

# Input
if prompt := st.chat_input("Ask me something..."):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
        st.stop()
        
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Init Model
        # Using the official ChatGoogleGenerativeAI wrapper
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True # Helps with some role errors
        )
        llm_with_tools = llm.bind_tools(tools)

        with st.chat_message("assistant"):
            # 1. First Call
            response_msg = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response_msg)
            
            # 2. Check for Tool Calls
            if response_msg.tool_calls:
                st.write(f"🛠️ *Thinking... (Calling: {', '.join([tc['name'] for tc in response_msg.tool_calls])})*")
                
                # Execute Tools loop
                for tool_call in response_msg.tool_calls:
                    selected_tool = tools_map[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    
                    tool_msg = ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"]
                    )
                    st.session_state.messages.append(tool_msg)

                # 3. Second Call (Get final answer)
                final_response = llm_with_tools.invoke(st.session_state.messages)
                st.markdown(final_response.content)
                st.session_state.messages.append(final_response)
            else:
                st.markdown(response_msg.content)

    except Exception as e:
        st.error(f"**Error:** {e}")
        st.info("💡 Tip: Use the 'Test Connection' button in the sidebar to see which models are available to your API Key.")
