import streamlit as st
import nbformat
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

st.set_page_config(page_title="Chat with your Notebook 📘", page_icon="📘")
st.title("Chat with Conchita Notebook 📘")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.info("⬅️ Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Load and Process Notebook
@st.cache_resource
def load_and_process_notebook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    text_content = ""
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            text_content += f"\n[MARKDOWN]\n{cell.source}\n"
        elif cell.cell_type == 'code':
            text_content += f"\n[CODE]\n{cell.source}\n"
    
    # Create Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text_content)
    documents = [Document(page_content=t, metadata={"source": file_path}) for t in splits]
    
    # Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

nb_path = "Conchita_EDEM_LLM_AI_App_for_Business_Marketing (1).ipynb"

if not os.path.exists(nb_path):
    st.error(f"Notebook file not found: {nb_path}")
    st.stop()

with st.spinner("Processing notebook... This may take a moment."):
    vectorstore = load_and_process_notebook(nb_path)
    retriever = vectorstore.as_retriever()

# LLM Setup
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask a question about the notebook..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt_input})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
