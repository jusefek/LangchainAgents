import langchain
print(f"LangChain version: {langchain.__version__}")
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
