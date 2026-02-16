import streamlit as st
import os

# Debugging Imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Page Config
st.set_page_config(page_title="Gemini Agent", page_icon="🤖")

# Sidebar for API Key
with st.sidebar:
    st.title("🤖 Configuration")
    
    # Debug info
    import langchain
    st.caption(f"LangChain Version: {langchain.__version__}")
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
        if not api_key:
            st.warning("Please enter your Google API Key to continue.")

# --- Custom Tools ---
@tool
def calculate_word_length(word: str) -> int:
    """Calculates the length of a given word."""
    return len(word)

@tool
def power_calculation(base: float, exponent: float) -> float:
    """Calculates the power of a base raised to an exponent."""
    return base ** exponent

tools = [calculate_word_length, power_calculation]

# --- Chat Interface ---
st.title("🤖 Google LangChain Agent")

with st.expander("ℹ️ **What can this agent do?**", expanded=True):
    st.write("""
    I am an intelligent agent powered by Google Gemini. I can help you with:
    
    1.  **General Conversation**: Chat with me about any topic.
    2.  **Word Analysis**: Ask me to count the letters in a word.
        *   *Example:* "How many letters are there in 'Streamlit'?"
    3.  **Math**: Ask me to calculate powers of numbers.
        *   *Example:* "Calculate 2 raised to the power of 10."
    
    Just type your question below! 👇
    """)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Interaction Logic ---
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.error("Please provide a Google API Key in the sidebar.")
        st.stop()

    # --- Agent Setup ---
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0
        )

        # Create Agent
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use your tools to answer questions. If you don't need a tool, just answer."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Execute with Visualization
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Prepare chat history for the agent context
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude the current user message
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            response = agent_executor.invoke(
                {"input": prompt, "chat_history": chat_history},
                {"callbacks": [st_callback]} 
            )
            
            output = response["output"]
            st.markdown(output)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": output})

    except Exception as e:
        st.error(f"An error occurred: {e}")
