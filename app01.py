import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Content Generator 🤖", page_icon="🤖")
st.title("Content generator 🤖")

# Sidebar for API Key
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

## Connection with the LLM
id_model = "llama-3.3-70b-versatile"
try:
    llm = ChatGroq(
        api_key=api_key,
        model=id_model,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
except Exception as e:
    st.error(f"Error initializing Client: {e}")
    st.stop()

## Generation function
def llm_generate(llm, prompt):
  template = ChatPromptTemplate.from_messages([
      ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
      ("human", "{prompt}"),
  ])

  chain = template | llm | StrOutputParser()

  res = chain.invoke({"prompt": prompt})
  return res

topic = st.text_input("Topic:", placeholder="e.g., nutrition, mental health, routine check-ups, self-care tips, etc.")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
tone = st.selectbox("Message tone:", ['Normal', 'Informative', 'Inspiring', 'Urgent', 'Informal'])
length = st.selectbox("Text length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target audience:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
cta = st.checkbox("Include CTA")
hashtags = st.checkbox("Return Hashtags")
keywords = st.text_area("Keywords (SEO):", placeholder="Example: wellness, preventive healthcare...")

if st.button("Generate Content"):
    if not topic:
         st.warning("Please enter a topic.")
    else:
        prompt = f"""
        Write an SEO-optimized text on the topic '{topic}'.
        Return only the final text in your response and don't put it inside quotes.
        - Platform where it will be published: {platform}.
        - Tone: {tone}.
        - Target audience: {audience}.
        - Length: {length}.
        - {"Include a clear Call to Action." if cta else "Do not include a Call to Action."}
        - {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
        {"- Keywords to include (for SEO): " + keywords if keywords else ""}
        """
        with st.spinner("Generating content..."):
            try:
                res = llm_generate(llm, prompt)
                st.markdown("### Generated Content:")
                st.markdown(res)
            except Exception as e:
                st.error(f"Error: {e}")
